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


def gmm_w2_distance(pred_means, pred_covs, pred_weights, prior_means, prior_covs, prior_weights):
    """
    Compute a discrete OT-based distance between two GMMs (with diagonal covariances)
    by matching each component with cost given by the Gaussian W2 distance.
    pred_means: (batch_size, n_components, latent_dim)
    pred_covs: (batch_size, n_components, latent_dim)
    pred_weights: (batch_size, n_components)

    prior_means: (n_components, latent_dim)
    prior_covs: (n_components, latent_dim)
    prior_weights: (n_components,)
    """
    batch_size = pred_means.shape[0]
    K = pred_means.shape[1]
    L = prior_means.shape[0]
    
    batch_distances = torch.zeros(batch_size, device=pred_means.device)

   # Process each sample in the batch
    for b in range(batch_size):
        # Get the GMM for this sample
        sample_pred_means = pred_means[b]  # (n_components, latent_dim)
        sample_pred_covs = pred_covs[b]    # (n_components, latent_dim)
        pred_sample_weights = pred_weights[b]  # (n_components,)
        
        # Build cost matrix for this sample
        

        cost_matrix = torch.zeros(K, L, device=pred_means.device)
        for k in range(K):
            for l in range(L):
                cost_matrix[k, l] = gaussian_w2_diagonal(
                    sample_pred_means[k], prior_means[l], 
                    sample_pred_covs[k], prior_covs[l]
                )
        
        # Convert to NumPy for OT solver
        # cost_np = cost_matrix.detach().cpu().numpy()
        # w1_np = pred_sample_weights.detach().cpu().numpy()
        # w2_np = prior_weights.detach().cpu().numpy()
        
        # Compute OT cost for this sample
        # dist = ot.emd2(w1_np, w2_np, cost_np)
        dist = ot.emd2(pred_sample_weights, prior_weights, cost_matrix)
        batch_distances[b] = dist
        # total_dist += dist

    # return torch.tensor(total_dist/batch_size, dtype=pred_means.dtype, device=pred_means.device)
    return torch.mean(batch_distances)