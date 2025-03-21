import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

class GMMVAE(nn.Module):
    def __init__(self, 
                 input_dim: int = 784, 
                 latent_dim: int = 2, 
                 n_components: int = 10, 
                 components_init: str = "circle", 
                 circle_radius: float = 3.0, 
                 to_predict: list = ["weights", "means", "covs"], 
                 prior_cov: float = 1.0):
        super().__init__()
        assert components_init in ["circle", "random"], "Components_init must be either circle or random"
        
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.input_dim = input_dim
        self.to_predict = to_predict

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # One per gaussian of the gmm
        if "means" in to_predict: self.means = nn.Linear(256, latent_dim * n_components)
        if "covs" in to_predict: self.log_vars = nn.Linear(256, latent_dim * n_components)
        if "weights" in to_predict: self.mixture_weights = nn.Linear(256, n_components)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
        
        # ==> Initialize priors
        prior_means = torch.zeros(n_components, latent_dim)
        if components_init == "circle":
            for i in range(n_components):
                angle = 2 * np.pi * i / n_components
                prior_means[i, 0] = circle_radius * np.cos(angle)
                prior_means[i, 1] = circle_radius * np.sin(angle)
        else:
            prior_means = torch.randn(n_components, latent_dim)

        self.register_buffer('prior_means', prior_means)
        self.register_buffer('prior_covs', torch.ones(n_components, latent_dim)*prior_cov)
        self.register_buffer('prior_weights', torch.ones(n_components) / n_components)
    

    def encode(self, x):
        h = self.encoder(x)
        batch_size = x.size(0)
        if "means" in self.to_predict:
            means = rearrange(self.means(h), "b (c l) -> b c l", c=self.n_components, l=self.latent_dim)
        else:
            means = repeat(self.prior_means, "c l -> b c l", b=batch_size)

        if "covs" in self.to_predict:
            log_vars = rearrange(self.log_vars(h), "b (c l) -> b c l", c=self.n_components, l=self.latent_dim)
        else:
            log_vars = repeat(torch.log(self.prior_covs), "c l -> b c l", b=batch_size) # !!! covs -> logvar
        
        if "weights" in self.to_predict:
            weights = torch.softmax(self.mixture_weights(h), dim=1)
        else:
            weights = repeat(self.prior_weights, "c -> b c", b=batch_size)
        return means, log_vars, weights
    
    def decode(self, z):
        return self.decoder(z)
    
    def gumbel_softmax_sample(self, weights, tau, eps=1e-10):
        u = torch.rand_like(weights) # (batch_size, n_components)
        gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
        return torch.softmax((torch.log(weights) + gumbel_noise) / tau, dim=-1)

    def reparameterize_gumbel(self, means, log_vars, weights, tau):
        """
        means: (batch_size, n_components, latent_dim)
        log_vars: (batch_size, n_components, latent_dim)
        weights: (batch_size, n_components)
        tau: Temperature for Gumbel-Softmax.
        """
        soft_assignments = self.gumbel_softmax_sample(weights, tau=tau)  # (batch_size, n_components)

        # Compute the weighted mean and log variance
        selected_means = torch.sum(soft_assignments.unsqueeze(-1) * means, dim=1)  # (batch_size, latent_dim)
        selected_log_vars = torch.sum(soft_assignments.unsqueeze(-1) * log_vars, dim=1)  # (batch_size, latent_dim)

        # Standard reparameterization trick
        std = torch.exp(0.5 * selected_log_vars)
        eps = torch.randn_like(std)
        z = selected_means + eps * std  # (batch_size, latent_dim)

        return z, soft_assignments

    def reparameterize(self, means, log_vars, weights):
        """
        means: (batch_size, n_components, latent_dim)
        log_vars: (batch_size, n_components, latent_dim)
        weights: (batch_size, n_components)
        """
        batch_size = means.size(0)
        # One component per sample
        components = torch.argmax(weights, dim=1) # argmax since it's not intended for training
        # Means and logvars for selected components
        batch_indices = torch.arange(batch_size, device=means.device)
        selected_means = means[batch_indices, components] # (batch_size, latent_dim)
        selected_log_vars = log_vars[batch_indices, components] # (batch_size, latent_dim)
        
        # Reparameterize
        std = torch.exp(0.5 * selected_log_vars)
        eps = torch.randn_like(std)
        z = selected_means + eps * std
        return z, components
    
    def project_to_latent(self, x):
        if len(x.shape) > 2:
            x = rearrange(x, "b c h w -> b (c h w)")
        means, log_vars, weights = self.encode(x)
        z, components = self.reparameterize(means, log_vars, weights)
        return z, means, log_vars, weights, components
    
    def forward(self, x, tau):
        """
        x: (batch_size, 1, 28, 28)
        """
        means, log_vars, weights = self.encode(rearrange(x, "b c h w -> b (c h w)"))
        z, soft_assignments = self.reparameterize_gumbel(means, log_vars, weights, tau)
        x_recon = self.decode(z)
        return x_recon, means, log_vars, weights, z, soft_assignments