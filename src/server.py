import torch
import copy
from torch.utils.data import DataLoader

# Import local modules
from client import SimpleCNN
from defense import AnomalyDetector, TrustManager
from aggregation import RobustAggregator

class AggregationServer:
    """Central orchestrator managing global state and the defensive pipeline."""
    def __init__(self, testset, device='cpu'):
        self.device = device
        self.global_model = SimpleCNN().to(self.device)
        self.testloader = DataLoader(testset, batch_size=64, shuffle=False)
        
    def evaluate_global_model(self):
        """Tests the global model's accuracy on the holdout test dataset."""
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100.0 * correct / total

    def apply_update(self, aggregated_update):
        """Applies the verified, aggregated delta weights to the global model."""
        state_dict = self.global_model.state_dict()
        for key in state_dict.keys():
            state_dict[key] += aggregated_update[key]
        self.global_model.load_state_dict(state_dict)

    def trigger_round(self, clients, aggregator, anomaly_detector=None, trust_manager=None):
        """Executes a single federated learning round with optional defensive filtering."""
        updates = {}
        
        # 1. Local Training: Broadcast global model and collect raw updates
        for client in clients:
            updates[client.client_id] = client.execute_local_sgd(self.global_model)
            
        # 2. Defense Pipeline: Statistical filtering and trust evaluation
        verified_updates = updates
        flags = {client.client_id: False for client in clients}
        
        if anomaly_detector and trust_manager:
            flags, norms = anomaly_detector.flag_outliers(updates)
            verified_updates = trust_manager.filter_updates(updates, flags)
            
            # Failsafe: If all updates are dropped, skip aggregation for this round
            if not verified_updates:
                return flags
                
        # 3. Robust Aggregation: Merge verified parameters
        aggregated_update = aggregator.aggregate(verified_updates)
        
        # 4. Global Update: Apply hardened weights
        self.apply_update(aggregated_update)
        
        return flags