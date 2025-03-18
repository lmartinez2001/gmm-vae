import torch
import ot

def gaussian_w2_diagonal(m0, m1, cov0, cov1):
    """
    Compute the 2-Wasserstein distance between two Gaussians in latent space with diagonal covariances.
    Formula: ||m0 - m1||^2 + sum_i (cov0[i] + cov1[i] - 2 * sqrt(cov0[i] * cov1[i]))

    """
    diff = torch.sum((m0 - m1) ** 2)
    trace_term = torch.sum(cov0 + cov1 - 2 * torch.sqrt(cov0 * cov1 + 1e-8))
    return diff + trace_term


def gmm_w2_distance(means1, covs1, weights1, means2, covs2, weights2):
    """
    Compute a discrete OT-based distance between two GMMs (with diagonal covariances)
    by matching each component with cost given by the Gaussian W2 distance.
    means1: (batch_size, n_components, latent_dim)
    covs1: (batch_size, n_components, latent_dim)
    weights1: (batch_size, n_components)

    means2: (n_components, latent_dim)
    covs2: (n_components, latent_dim)
    weights2: (n_components,)
    """
    batch_size = means1.shape[0]
    total_dist = 0.0

   # Process each sample in the batch
    for b in range(batch_size):
        # Get the GMM for this sample
        sample_means1 = means1[b]  # (n_components, latent_dim)
        sample_covs1 = covs1[b]    # (n_components, latent_dim)
        sample_weights1 = weights1[b]  # (n_components,)
        
        # Build cost matrix for this sample
        K = sample_means1.shape[0]
        L = means2.shape[0]
        cost_matrix = torch.zeros(K, L, device=means1.device)
        
        for k in range(K):
            for l in range(L):
                cost_matrix[k, l] = gaussian_w2_diagonal(
                    sample_means1[k], means2[l], 
                    sample_covs1[k], covs2[l]
                )
        
        # Convert to NumPy for OT solver
        cost_np = cost_matrix.detach().cpu().numpy()
        w1_np = sample_weights1.detach().cpu().numpy()
        w2_np = weights2.detach().cpu().numpy()
        
        # Compute OT cost for this sample
        dist = ot.emd2(w1_np, w2_np, cost_np)
        total_dist += dist

    return torch.tensor(total_dist/batch_size, dtype=means1.dtype, device=means1.device)