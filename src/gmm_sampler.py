import torch
import torch.nn.functional as F

class DifferentiableGMMSampler:
    
    def sample_gumbel(self, shape, eps=1e-10):
        u = torch.rand(shape)
        return -torch.log(-torch.log(u + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature):
        gumbel_noise = self.sample_gumbel(logits.shape)
        gumbel_logits = (logits + gumbel_noise) / temperature
        y = F.softmax(gumbel_logits, dim=-1)
        return y
    
    def sample(self, weights_logits, means, variances, temperature=1.0, hard=False, n_samples=1):
        batch_size, n_components = weights_logits.shape
        _, _, feature_dim = means.shape
        
        weights = F.softmax(weights_logits, dim=-1)
        
        stds = torch.sqrt(variances)
        
        weights = weights.unsqueeze(1).expand(batch_size, n_samples, n_components)
        means = means.unsqueeze(1).expand(batch_size, n_samples, n_components, feature_dim)
        stds = stds.unsqueeze(1).expand(batch_size, n_samples, n_components, feature_dim)
        
        log_weights = torch.log(weights + 1e-10) 
        component_probs = self.gumbel_softmax_sample(log_weights, temperature) # Can also take raw logits as input

        if hard:
            indices = torch.argmax(component_probs, dim=-1)
            component_probs_hard = F.one_hot(indices, n_components).float()
            component_probs = (component_probs_hard - component_probs).detach() + component_probs
        
        epsilon = torch.randn(batch_size, n_samples, 1, feature_dim)

        component_samples = means + stds * epsilon
        
        component_probs = component_probs.unsqueeze(-1)  # (batch, samples, components, 1)
        weighted_samples = component_probs * component_samples
        
        samples = weighted_samples.sum(dim=2)  # (batch, samples, feature_dim)
        
        return samples