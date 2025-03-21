import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from src.gmm_sampler import DifferentiableGMMSampler

class SimpleVAE(torch.nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_components: int, distribution: str = "normal"):
        """
        hidden_dim: dimension of the hidden layers
        latent_dim: dimension of the latent representation
        activation: callable activation function
        distribution: string either `normal` or `GMM`, indicates which distribution to use
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.distribution = distribution
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.gmm_sampler = DifferentiableGMMSampler()
        
        
        self.fc_e0 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc_e1 = nn.Linear(hidden_dim * 2, hidden_dim)

        if self.distribution == "normal":
            self.fc_mean = nn.Linear(hidden_dim, latent_dim)
            self.fc_var =  nn.Linear(hidden_dim, latent_dim)
        elif self.distribution == "gmm":
            self.fc_means = nn.Linear(hidden_dim, n_components * latent_dim)
            self.fc_vars = nn.Linear(hidden_dim, n_components * latent_dim)
            self.fc_weights = nn.Linear(hidden_dim, n_components)


        self.fc_d0 = nn.Linear(latent_dim, hidden_dim)
        self.fc_d1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc_logits = nn.Linear(hidden_dim * 2, input_dim)

    def encode(self, x):
        x = F.relu(self.fc_e0(x))
        x = F.relu(self.fc_e1(x))
        
        if self.distribution == "normal":
            z_mean = self.fc_mean(x)
            z_var = F.softplus(self.fc_var(x))
            return z_mean, z_var
        elif self.distribution == "gmm":
            means = self.fc_means(x)
            means = rearrange(means, "b (c l) -> b c l", c=self.n_components, l=self.latent_dim)

            log_vars = self.fc_vars(x)
            log_vars = rearrange(log_vars, "b (c l) -> b c l", c=self.n_components, l=self.latent_dim)
            variances = torch.exp(log_vars)
            weights_logits = self.fc_weights(x) # (batch_size, n_components)
            return means, variances, weights_logits
        else:
            raise NotImplemented
        
    def decode(self, z):
        x = F.relu(self.fc_d0(z))
        x = F.relu(self.fc_d1(x))
        x = self.fc_logits(x)
        return x
        
    def reparameterize(self, *args):
        if self.distribution == "normal":
            z_mean, z_var = args
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var)) # prior
            return p_z, q_z
        elif self.distribution == "gmm":
            means, variances, weights_logits, temperature = args
            z = self.gmm_sampler.sample(weights_logits=weights_logits, 
                                        means=means, 
                                        variances=variances, 
                                        temperature=temperature, 
                                        hard=False, 
                                        n_samples=1)
            return z
        else:
            raise NotImplemented
        
    def forward(self, x, temperature=None):
        if self.distribution == "normal":
            z_mean, z_var = self.encode(x)
            q_z, p_z = self.reparameterize(z_mean, z_var)
            z = q_z.rsample()
            out = self.decode(z)
            return (z_mean, z_var), (q_z, p_z), z, out
        elif self.distribution  == "gmm":
            means, variances, weights_logits = self.encode(x)
            z = self.reparameterize(means, variances, weights_logits, temperature).squeeze(1)
            out = self.decode(z)
            return means, variances, weights_logits, z, out
        else:
            raise NotImplemented