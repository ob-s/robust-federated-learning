import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

def get_dataset(dataset_name='mnist', data_dir='../data'):
    """Downloads and applies standard transforms to the selected benchmark dataset."""
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        testset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError("Dataset must be 'mnist' or 'cifar10'")
        
    return trainset, testset

def partition_data(trainset, num_clients, iid=True, alpha=0.5):
    """
    Partitions the training dataset across clients.
    - iid: Uniformly distributes data.
    - non-iid: Uses Dirichlet distribution (alpha) to create label imbalances.
    """
    num_items = int(len(trainset) / num_clients)
    client_datasets = {}
    
    if iid:
        all_idxs = np.random.permutation(len(trainset))
        for i in range(num_clients):
            client_datasets[i] = all_idxs[i * num_items : (i + 1) * num_items]
    else:
        # Pathological non-IID partitioning using Dirichlet distribution
        labels = np.array(trainset.targets)
        num_classes = len(np.unique(labels))
        client_datasets = {i: np.array([], dtype='int64') for i in range(num_clients)}
        idxs = np.arange(len(trainset))
        
        for k in range(num_classes):
            idx_k = idxs[labels == k]
            np.random.shuffle(idx_k)
            # Create fractional splits for each client for this class
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < num_items) for p, idx_j in zip(proportions, client_datasets.values())])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_k_split = np.split(idx_k, proportions)
            for i in range(num_clients):
                client_datasets[i] = np.concatenate((client_datasets[i], idx_k_split[i]))

    # Convert index arrays back into PyTorch Subset objects
    return {i: Subset(trainset, indices) for i, indices in client_datasets.items()}