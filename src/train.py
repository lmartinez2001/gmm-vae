
import os
import uuid
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import torchvision.datasets as datasets

from gmmvae import GMMVAE
from gmmot import gmm_w2_distance

from einops import rearrange
from utils import *

torch.manual_seed(0)

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.family'] = "serif"


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer, epoch: int, device: str, beta: float):
    model.train()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_mw2_loss = 0

    for batch_idx, (data, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, means, log_vars, weights, z, components = model(data)
        
        recon_loss = nn.BCELoss(reduction="mean")(recon_batch, rearrange(data, "b c h w -> b (c h w)"))
        
        covs = torch.exp(log_vars)
        
        # MW2 loss between encoded GMM and prior GMM
        mw2_loss = gmm_w2_distance(means, # (batch_size, n_components, latent_dim)
                                   covs, # (batch_size, n_components, latent_dim)
                                   weights, # (batch_size, n_components)
                                   model.prior_means, # (n_components, latent_dim)
                                   model.prior_covs.view(model.n_components, -1), # (n_components, latent_dim)
                                   model.prior_weights) # (n_components,)
        
        # Total loss
        loss = recon_loss + beta * mw2_loss

        loss.backward()
        epoch_loss += loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_mw2_loss += mw2_loss.item()

        optimizer.step()
        
    avg_loss = epoch_loss / len(train_loader.dataset)
    avg_recon = epoch_recon_loss / len(train_loader.dataset)
    avg_mw2 = epoch_mw2_loss / len(train_loader.dataset)

    return avg_loss, avg_recon, avg_mw2


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # ==> Parameters
    input_dim = 784
    latent_dim = 2
    n_components = 10
    n_epochs = 40
    lr = 0.001
    batch_size = 256
    beta = 1.0
    sample_interval = 5
    subset_ratio = 3

    save_root = os.path.join("results", str(uuid.uuid1()))
    os.makedirs(save_root)

    save_params(root_path=save_root,
                input_dim=input_dim,
                latent_dim=latent_dim,
                n_components=n_components,
                beta=beta,
                training_samples=60000//subset_ratio
                )
    
    transform = T.Compose([
        T.ToTensor()
    ])
    

    train_dataset, test_dataset = load_MNIST("./data", subset_ratio=3)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GMMVAE(input_dim=input_dim,
                   latent_dim=latent_dim,
                   n_components=n_components)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    all_losses = {
        "total": [],
        "recon": [],
        "mw2": []
    }
    
    with tqdm(total=n_epochs, desc="Training", unit="epoch") as pbar:
        for epoch in range(n_epochs):
            epoch_loss, epoch_recon, epoch_mw2 = train_epoch(model, train_loader, optimizer, epoch, device, beta=beta)
            all_losses["total"].append(epoch_loss)
            all_losses["recon"].append(epoch_recon)
            all_losses["mw2"].append(epoch_mw2)
            
            # ==> Sampling to check progression
            if (epoch % sample_interval == 0) or (epoch == n_epochs-1):
                visualize(model=model,
                          test_loader=test_loader,
                          save_path=save_root, 
                          device=device, 
                          epoch=epoch)
                save_model(save_path=save_root, model=model, epoch=epoch)
            pbar.set_postfix({"Loss": f"{epoch_loss:.2f}"})
            pbar.update(1)

    # Save losses
    save_losses(save_root, all_losses)

    print(f"Training completed. Results saved to {save_root}")