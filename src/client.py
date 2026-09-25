import torch
import torch.nn as nn
import torch.optim as optim
import copy

class SimpleCNN(nn.Module):
    """A lightweight CNN architecture that works for both MNIST and CIFAR-10."""
    def __init__(self, in_channels=1, num_classes=10, dim=28):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        linear_dim = 64 * (dim // 4) * (dim // 4)
        self.classifier = nn.Sequential(
            nn.Linear(linear_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ClientNode:
    """Simulates an honest edge device performing standard local training."""
    def __init__(self, client_id, dataset, device='cpu', lr=0.01, epochs=1, batch_size=32):
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        
    def execute_local_sgd(self, global_model):
        """Standard localized training on private data to compute authentic updates."""
        model = copy.deepcopy(global_model).to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(self.epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
        # Calculate delta (local weights - global weights)
        update = {}
        for key in global_model.state_dict().keys():
            update[key] = model.state_dict()[key] - global_model.state_dict()[key]
            
        return update

class MaliciousClient(ClientNode):
    """Simulates a compromised edge device executing model poisoning attacks."""
    def __init__(self, client_id, dataset, device='cpu', lr=0.01, epochs=1, batch_size=32, attack_type='byzantine', scaling_factor=10.0):
        super().__init__(client_id, dataset, device, lr, epochs, batch_size)
        self.attack_type = attack_type
        self.scaling_factor = scaling_factor
        
    def poison_weights(self, update):
        """Manipulates gradients to execute Byzantine scaling or targeted payloads."""
        poisoned_update = {}
        for key, value in update.items():
            if self.attack_type == 'byzantine':
                # Scale the authentic weights aggressively to disrupt global aggregation
                poisoned_update[key] = value * self.scaling_factor
            elif self.attack_type == 'random':
                # Inject complete noise to corrupt the model state
                poisoned_update[key] = torch.randn_like(value) * self.scaling_factor
            else:
                poisoned_update[key] = value
        return poisoned_update
        
    def execute_local_sgd(self, global_model):
        """Overrides honest training to inject adversarial payloads."""
        # 1. Perform standard SGD to get normal update geometry
        authentic_update = super().execute_local_sgd(global_model)
        
        # 2. Activate poisoning engine to construct adversarial payload
        malicious_update = self.poison_weights(authentic_update)
        
        return malicious_update