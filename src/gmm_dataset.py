from torch.utils.data import Dataset
import torch
from src.data import load_toy_gmm 

class GMMDataset(Dataset):
    def __init__(self, 
                 n_gaussian=5, 
                 n_samples=1000, 
                 cov=0.2, 
                 radius=2, 
                 transform=None, 
                 dim=100,
                 noise_std=0.1):

        self.samples, self.labels, self.means, self.covs, self.weights = load_toy_gmm(
            n_gaussian=n_gaussian, 
            n_samples=n_samples, 
            cov=cov, 
            radius=radius
        )
        self.means = torch.tensor(self.means).float()
        self.transform = transform
        self.samples = torch.tensor(self.samples).float()
        self.labels = torch.tensor(self.labels).long()

        # samples (n_samples, 2)
        self.centers = 4 * torch.rand(dim, 2) - 2  # [-2,2] # 100, 2
        self.projected_samples = torch.exp(-torch.sum((self.samples[:, None, :] - self.centers[None, :, :])**2, dim=-1)).float() + noise_std * torch.randn(n_samples, dim)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        projected_sample = self.projected_samples[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, projected_sample, label