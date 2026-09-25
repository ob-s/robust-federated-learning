import torch

class RobustAggregator:
    """Implements baseline and Byzantine-resilient aggregation rules."""
    def __init__(self, aggregation_rule='fedavg', trim_percentage=0.1, num_byzantine=1):
        self.aggregation_rule = aggregation_rule.lower()
        self.trim_percentage = trim_percentage
        self.num_byzantine = num_byzantine

    def _flatten_update(self, update):
        """Flattens a dictionary of parameter tensors into a single 1D vector."""
        return torch.cat([param.contiguous().view(-1) for param in update.values()])

    def fedavg(self, updates):
        """Standard Federated Averaging baseline."""
        num_clients = len(updates)
        avg_update = {}
        first_client = next(iter(updates.values()))
        
        for key in first_client.keys():
            stacked = torch.stack([client_update[key].float() for client_update in updates.values()])
            avg_update[key] = torch.mean(stacked, dim=0)
        return avg_update

    def coordinate_median(self, updates):
        """Coordinate-wise median aggregation."""
        med_update = {}
        first_client = next(iter(updates.values()))
        
        for key in first_client.keys():
            stacked = torch.stack([client_update[key].float() for client_update in updates.values()])
            med_update[key], _ = torch.median(stacked, dim=0)
        return med_update

    def trimmed_mean(self, updates):
        """Coordinate-wise trimmed mean aggregation."""
        num_clients = len(updates)
        trim_count = int(self.trim_percentage * num_clients)
        trimmed_update = {}
        first_client = next(iter(updates.values()))
        
        for key in first_client.keys():
            stacked = torch.stack([client_update[key].float() for client_update in updates.values()])
            sorted_tensor, _ = torch.sort(stacked, dim=0)
            
            # Slice away the extreme high and low coordinates
            if trim_count > 0:
                retained = sorted_tensor[trim_count : num_clients - trim_count]
            else:
                retained = sorted_tensor
            trimmed_update[key] = torch.mean(retained, dim=0)
        return trimmed_update

    def krum(self, updates):
        """
        Krum Byzantine-resilient aggregation rule.
        Selects the client update closest to its n - f - 2 nearest neighbors.
        """
        client_ids = list(updates.keys())
        n = len(client_ids)
        f = self.num_byzantine
        
        # Upper bound constraint verification (2f + 2 < n)[cite: 1]
        neighbors_to_sum = max(1, n - f - 2)
        
        flat_vectors = [self._flatten_update(updates[cid]).float() for cid in client_ids]
        scores = []
        
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = torch.norm(flat_vectors[i] - flat_vectors[j]) ** 2
                    distances.append(dist.item())
            distances.sort()
            # Sum the squared distances to the nearest neighbors
            score = sum(distances[:neighbors_to_sum])
            scores.append(score)
            
        selected_idx = scores.index(min(scores))
        selected_client_id = client_ids[selected_idx]
        return updates[selected_client_id]

    def aggregate(self, updates):
        """Pipes updates into the selected consensus strategy."""
        if not updates:
            raise ValueError("Cannot aggregate an empty update pool.")
        if len(updates) == 1:
            return next(iter(updates.values()))

        if self.aggregation_rule == 'fedavg':
            return self.fedavg(updates)
        elif self.aggregation_rule == 'median':
            return self.coordinate_median(updates)
        elif self.aggregation_rule == 'trimmed_mean':
            return self.trimmed_mean(updates)
        elif self.aggregation_rule == 'krum':
            return self.krum(updates)
        else:
            raise ValueError(f"Unknown aggregation rule: {self.aggregation_rule}")