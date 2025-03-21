import os
import json

import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader



from torchvision.utils import make_grid


# ==> Utily functions to save files
def save_params(root_path: str, **kwargs):
    with open(os.path.join(root_path, "model_params.json"), "w") as f:
        json.dump(kwargs, f)
    print(f"Model params saved in {root_path}")

def save_losses(root_path: str, losses: dict):
    with open(os.path.join(root_path, "losses.pkl"), "wb") as f:
        pickle.dump(losses, f)
    print(f"Losses saved in {root_path}")

def load_params(root_path: str):
    with open(os.path.join(root_path, "model_params.json"), "r") as f:
        model_params = json.load(f)
    print(f"Model params loaded")
    return model_params

def load_ckpt(root_path: str, epoch: int):
    return torch.load(os.path.join(root_path, f"model_epoch{epoch}.pt"))

def load_losses(root_path: str):
    with open(os.path.join(root_path, "losses.pkl"), "rb") as f:
        losses = pickle.load(f)
    return losses

def save_model(save_path: str, model: nn.Module, epoch: int):
    model_path = os.path.join(save_path, f"model_epoch{epoch}.pt")
    torch.save(model.state_dict(), model_path)


def visualize(model: nn.Module, save_path: str, test_loader: DataLoader, device: str, epoch: int):
    model.eval()

    # ==> Generate samples
    with torch.no_grad():
        samples = model.decode(model.prior_means)
    samples = samples.view(-1, 1, 28, 28).cpu()
    grid = make_grid(samples, nrow=5, padding=2, normalize=True).permute(1, 2, 0)
    plt.figure()
    plt.axis("off")
    plt.imshow(grid)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"means_samples_epoch{epoch}.png"), bbox_inches='tight', transparent=True, pad_inches=0.0, dpi=300)
    plt.close()

    # ==> Project test in latent space
    all_z = []
    all_labels = []
    for data, label in test_loader:
        data = data.to(device) 
        with torch.no_grad():
            z, _, _, _, _ = model.project_to_latent(data)
            all_z.append(z.cpu())
            all_labels.append(label)

    to_show = torch.cat(all_z, dim=0)
    c = torch.cat(all_labels, dim=0)
    # One with no axis
    plt.figure(figsize=(10,10))
    plt.scatter(to_show[:,1], to_show[:,0], c=c, cmap="tab10", alpha=0.3, s=1)
    plt.axis("off")
    plt.savefig(os.path.join(save_path, f"test_set_latent_epoch{epoch}.png"), bbox_inches='tight', transparent=True, pad_inches=0.0, dpi=300)
    plt.close()

    # One with axis
    plt.figure(figsize=(10,10))
    plt.scatter(to_show[:,1], to_show[:,0], c=c, cmap="tab10", alpha=0.3, s=1)
    plt.savefig(os.path.join(save_path, f"test_set_latent_epoch{epoch}_axis.png"), bbox_inches='tight', transparent=True, pad_inches=0.0, dpi=300)
    plt.close()