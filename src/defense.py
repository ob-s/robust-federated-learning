import torch
import numpy as np

class AnomalyDetector:
    """Evaluates incoming tensors to identify statistical outliers."""
    def __init__(self, std_threshold=2.0):
        # Flags updates deviating >2 standard deviations from the median
        self.std_threshold = std_threshold
        
    def compute_euclidean_norm(self, update):
        """Computes the L2 norm (Euclidean distance) of a client's gradient update."""
        squared_sum = 0.0
        for key, tensor in update.items():
            squared_sum += torch.sum(tensor ** 2).item()
        return np.sqrt(squared_sum)

    def flag_outliers(self, updates):
        """
        Analyzes the tensor norms of all submissions. Returns a dictionary 
        flagging updates that exceed the dynamic threshold.
        """
        norms = {client_id: self.compute_euclidean_norm(update) for client_id, update in updates.items()}
        norm_values = list(norms.values())
        
        batch_median = np.median(norm_values)
        # Add small epsilon to avoid division by zero if all updates are identical
        batch_std = np.std(norm_values) if np.std(norm_values) > 0 else 1e-5 
        
        flags = {}
        for client_id, norm in norms.items():
            # Calculate how far the norm is from the batch median
            deviation = abs(norm - batch_median) / batch_std
            is_outlier = deviation > self.std_threshold
            flags[client_id] = is_outlier
            
        return flags, norms

class TrustManager:
    """Evaluates historical behavior and handles dynamic client penalty distributions."""
    def __init__(self, client_ids, initial_trust=1.0, penalty=0.25):
        self.trust_registry = {client_id: initial_trust for client_id in client_ids}
        self.penalty = penalty
        
    def penalize_trust_score(self, flags):
        """Applies a weighted penalty to the trust score of flagged clients."""
        for client_id, is_malicious in flags.items():
            if is_malicious:
                # Diminish future influence by dropping the score, floor at 0.0
                self.trust_registry[client_id] = max(0.0, self.trust_registry[client_id] - self.penalty)
        return self.trust_registry
        
    def filter_updates(self, updates, flags, min_trust_threshold=0.5):
        """
        Pipes verified updates to the aggregator. Drops updates if they trigger 
        an anomaly flag in the current round OR have a poor historical trust score.
        """
        verified_updates = {}
        self.penalize_trust_score(flags)
        
        for client_id, update in updates.items():
            # Only keep updates that are clean AND from trustworthy clients
            if not flags[client_id] and self.trust_registry[client_id] >= min_trust_threshold:
                verified_updates[client_id] = update
                
        return verified_updates