import numpy as np
from scipy.stats import multivariate_normal

import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import Subset

# ==> Toy datasets
def load_toy_gmm(n_gaussian, n_samples, cov, radius=2):
    theta = np.linspace(0, 2 * np.pi, n_gaussian, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    means = np.column_stack((x, y))
    
    covs = [np.eye(2) * cov for _ in range(n_gaussian)]
    weights = np.ones(n_gaussian) / n_gaussian 
    
    samples, labels = sample_gmm(n_samples, means, covs, weights)
    
    return samples, labels, means, covs, weights

def sample_gmm(n_samples, means, covariances, weights):
    n_components = len(means)
    labels = np.random.choice(n_components, size=n_samples, p=weights)
    samples = np.zeros((n_samples, 2))
    
    for i in range(n_components):
        component_indices = np.where(labels == i)[0]
        n_component_samples = len(component_indices)
        if n_component_samples > 0:
            component_samples = multivariate_normal.rvs(mean=means[i], 
                                                        cov=covariances[i], 
                                                        size=n_component_samples)
            samples[component_indices] = component_samples
    return samples, labels


# ==> MNIST dataset
def load_MNIST(root: str, subset_ratio: int = 1):
    transform = T.Compose([T.ToTensor()])
    # Load train
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    if subset_ratio > 1:
        train_size = len(train_dataset)
        n_train_samples = train_size // subset_ratio
        train_indices = np.random.choice(train_size, n_train_samples, replace=False)
        train_dataset = Subset(train_dataset, train_indices)
    # Load test
    test_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    return train_dataset, test_dataset
