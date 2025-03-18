import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from scipy.optimize import linear_sum_assignment
from einops import rearrange

class GMMVAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=2, n_components=10):
        super(GMMVAE, self).__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.input_dim = input_dim
        
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # One per gaussian of the gmm
        self.means = nn.Linear(256, latent_dim * n_components)
        self.log_vars = nn.Linear(256, latent_dim * n_components)
        self.mixture_weights = nn.Linear(256, n_components)
        
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
        
        # Prior with uniform probability for each gaussian
        self.register_buffer('prior_means', torch.zeros(n_components, latent_dim))
        self.register_buffer('prior_covs', torch.ones(n_components, latent_dim))
        self.register_buffer('prior_weights', torch.ones(n_components) / n_components)
    
    def encode(self, x):
        h = self.encoder(x)
        means = rearrange(self.means(h), "b (c l) -> b c l", c=self.n_components, l=self.latent_dim)
        log_vars = rearrange(self.log_vars(h), "b (c l) -> b c l", c=self.n_components, l=self.latent_dim)
        weights = torch.softmax(self.mixture_weights(h), dim=1) # sum up to 1
        return means, log_vars, weights
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, means, log_vars, weights):
        """
        means: (batch_size, n_components, latent_dim)
        log_vars: (batch_size, n_components, latent_dim)
        weights: (batch_size, n_components)
        """
        # One component per sample
        components = torch.multinomial(weights, 1).squeeze()
        
        # Means and logvars for selected components
        batch_indices = torch.arange(means.size(0))
        selected_means = means[batch_indices, components]
        selected_log_vars = log_vars[batch_indices, components]
        
        # Reparameterize
        std = torch.exp(0.5 * selected_log_vars)
        eps = torch.randn_like(std)
        z = selected_means + eps * std
        return z, components
    
    def forward(self, x):
        """
        x: (batch_size, 28, 28)
        """
        means, log_vars, weights = self.encode(rearrange(x, "b c h w -> b (c h w)"))
        z, components = self.reparameterize(means, log_vars, weights)
        x_recon = self.decode(z)
        return x_recon, means, log_vars, weights, z, components

# # Compute Wasserstein-2 distance between Gaussians
# def w2_gaussian(mean1, cov1, mean2, cov2):
#     """Compute W2 distance between two Gaussians"""
#     mean_diff = mean1 - mean2
#     mean_term = torch.sum(mean_diff ** 2)
    
#     # For diagonal covariances (as in our case)
#     cov_term = torch.sum(cov1 + cov2 - 2 * torch.sqrt(cov1 * cov2))
    
#     return mean_term + cov_term

# # Compute MW2 distance between GMMs
# def compute_mw2(means1, covs1, weights1, means2, covs2, weights2):
#     """Compute MW2 distance between two GMMs"""
#     n1, d = means1.shape
#     n2, _ = means2.shape
    
#     # Compute pairwise W2 distances between all components
#     cost_matrix = torch.zeros(n1, n2)
#     for i in range(n1):
#         for j in range(n2):
#             cost_matrix[i, j] = w2_gaussian(means1[i], covs1[i], means2[j], covs2[j])
    
#     # Solve optimal transport problem
#     # Convert to numpy for linear_sum_assignment
#     cost_np = cost_matrix.detach().cpu().numpy()
#     weights1_np = weights1.detach().cpu().numpy()
#     weights2_np = weights2.detach().cpu().numpy()
    
#     # Solve the linear assignment problem
#     # Note: This is a simplified version for equal weights
#     # For general weights, we need to solve a linear program
#     row_ind, col_ind = linear_sum_assignment(cost_np)
    
#     # Compute final MW2 distance
#     mw2_dist = 0.0
#     for i, j in zip(row_ind, col_ind):
#         mw2_dist += min(weights1_np[i], weights2_np[j]) * cost_np[i, j]
    
#     return torch.tensor(mw2_dist, device=means1.device)


# # Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load MNIST dataset
# transform = transforms.Compose([transforms.ToTensor()])
# train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# # Initialize model and optimizer
# model = GMMVAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# # Train the model
# for epoch in range(1, 11):
#     train(model, train_loader, optimizer, epoch)