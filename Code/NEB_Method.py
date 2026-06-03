"""
Implements the NEB method on the score function (s_theta) and outputs a parameter sweep of NEB improved interpolations for many different learning rates and spring forces.
"""

import os
import math
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------
# UNetSD Components
# ------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timestep):
        device = timestep.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timestep[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_skip_conv=False):
        super().__init__()
        self.use_skip_conv = use_skip_conv
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if in_channels != out_channels or use_skip_conv:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        h = self.norm1(x)
        h = self.relu(h)
        h = self.conv1(h)
        time_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.norm2(h)
        h = self.relu(h)
        h = self.conv2(h)
        return h + self.skip_conv(x)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class UNetSD(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.downs = nn.ModuleList([
            ResidualBlock(base_channels, base_channels, time_emb_dim),
            ResidualBlock(base_channels, base_channels * 2, time_emb_dim, use_skip_conv=True),
            ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim, use_skip_conv=True),
        ])
        self.downsamples = nn.ModuleList([
            Downsample(base_channels),
            Downsample(base_channels * 2),
            Downsample(base_channels * 4),
        ])
        self.mid1 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.upsamples = nn.ModuleList([
            Upsample(base_channels * 4),
            Upsample(base_channels * 2),
            Upsample(base_channels),
        ])
        self.ups = nn.ModuleList([
            ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim, use_skip_conv=True),
            ResidualBlock(base_channels * 4, base_channels, time_emb_dim, use_skip_conv=True),
            ResidualBlock(base_channels * 2, base_channels, time_emb_dim, use_skip_conv=True),
        ])
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_relu = nn.ReLU()
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        residuals = []
        for block, down in zip(self.downs, self.downsamples):
            x = block(x, t_emb)
            residuals.append(x)
            x = down(x)
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)
        for upsample, block in zip(self.upsamples, self.ups):
            x = upsample(x)
            res = residuals.pop()
            x = torch.cat([x, res], dim=1)
            x = block(x, t_emb)
        x = self.out_norm(x)
        x = self.out_relu(x)
        x = self.out_conv(x)
        return x

# ------------------------
# VP Scheduler
# ------------------------
class VPScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        alphas_cumprod = self.alphas_cumprod.to(x0.device)
        if isinstance(t, int):
            t = torch.tensor([t], device=x0.device)
        sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

# ------------------------
# DDIM deterministic sampling & inversion
# ------------------------
def ddim_sample(model, scheduler, x_T, timesteps, eta=0.0):
    x_t = x_T
    device = x_t.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[t_prev]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        t_tensor = torch.tensor([t], device=device)
        epsilon_theta = model(x_t, t_tensor)
        x0_pred = (x_t - sqrt_one_minus_alpha_t * epsilon_theta) / sqrt_alpha_t
        sigma_t = eta * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        )
        sigma_t_val = sigma_t.item()
        noise = torch.randn_like(x_t) if sigma_t_val > 0 else torch.zeros_like(x_t)
        x_t = sqrt_alpha_t_prev * x0_pred + torch.sqrt(1 - alpha_t_prev - sigma_t_val**2) * epsilon_theta + sigma_t * noise
    return x_t

def ddim_invert(model, scheduler, x0, num_steps=1000):
    device = x0.device
    timesteps = torch.linspace(0, scheduler.num_timesteps - 1, num_steps, dtype=torch.long)
    x_t = x0.clone()
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        alpha_t = alphas_cumprod[t]
        alpha_t_next = alphas_cumprod[t_next]
        t_tensor = torch.tensor([t], device=device)
        epsilon_theta = model(x_t, t_tensor)
        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * epsilon_theta) / torch.sqrt(alpha_t)
        x_t = torch.sqrt(alpha_t_next) * x0_pred + torch.sqrt(1 - alpha_t_next) * epsilon_theta
    return x_t

# ------------------------
# Score function
# ------------------------
def score_fn(x, model, scheduler, t):
    device = x.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    if isinstance(t, int):
        t_tensor = torch.tensor([t], device=device)
        alpha_bar = alphas_cumprod[t]
    else:
        t_tensor = t
        alpha_bar = alphas_cumprod[t]
    epsilon_theta = model(x, t_tensor)
    score = - epsilon_theta / torch.sqrt(1 - alpha_bar)
    return score

# ------------------------
# DDIM forward & image helpers
# ------------------------
def load_image(path, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

@torch.no_grad()
def Phi(y, model, scheduler, num_steps=1000, eta=0.0):
    timesteps = torch.linspace(scheduler.num_timesteps-1, 0, num_steps, dtype=torch.long)
    x = ddim_sample(model, scheduler, y, timesteps, eta=eta)
    return x

# ------------------------
# NEB descent on the score function
# ------------------------
def neb_descent_on_score(curve, model, scheduler, t_idx,
                         n_iters=50, lr=0.01, k=1.0,
                         reparam_every=5):
    """
    Nudged Elastic Band descent using the diffusion score as the potential force.

    curve : [L, C, H, W] - chain of images in latent space (endpoints fixed)
    t_idx : diffusion timestep at which the score is evaluated
    lr    : step size
    k     : spring constant (equal spacing)
    reparam_every : how often to redistribute images equally along the path
    """
    L = curve.shape[0]
    curves = [curve.clone()]

    for it in range(n_iters):
        new_curve = curve.clone()
      
        # ---- NEB force update for interior images ----
        for i in range(1, L - 1):
            x_i = curve[i]                 # [C, H, W]
            prev = curve[i-1]
            nxt  = curve[i+1]

            # Tangent (simple forward-backward bisector)
            tau = (nxt - prev).flatten()
            tau = tau / (tau.norm() + 1e-8)

            # Score (potential force)
            s = score_fn(x_i.unsqueeze(0), model, scheduler, t_idx).flatten()

            # Perpendicular component of the force
            s_perp = s - (s @ tau) * tau

            # Spring force (equal spacing)
            d_prev = (x_i - prev).flatten().norm()
            d_next = (nxt - x_i).flatten().norm()
            f_spring = -k * (d_prev - d_next) * tau

            # Update
            dx = lr * (s_perp + f_spring)
            new_curve[i] = curve[i] + dx.view_as(x_i)

        # Fix endpoints
        new_curve[0] = curve[0]
        new_curve[-1] = curve[-1]

        curve = new_curve
        curves.append(curve.clone())

        # Progress print
        if it > 0:
            delta = (curve - curves[-2]).norm().item()

    return curves

@torch.no_grad()
def plot_curve_images(curves, Phi_fn, title="High-Confidence Curve"):
    n_curves = len(curves)
    L = curves[0].shape[0]
    fig, axes = plt.subplots(
        n_curves, L, figsize=(1.8 * L, 1.8 * n_curves)
    )
    if n_curves == 1:
        axes = axes[None, :]
    for i, curve in enumerate(curves):
        imgs = Phi_fn(curve)
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
        for j in range(L):
            axes[i, j].imshow(
                transforms.ToPILImage()(imgs[j].cpu())
            )
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(f"s={j}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("NEB_Curves.png")
    print("Saved NEB curve figure.")

@torch.no_grad()
def plot_sweep_summary(final_curves, lr_vals, k_vals, Phi_fn,
                       save_path="NEB_sweep_summary.png", dpi=100):
    """
    Create a single full-resolution figure: each subplot shows the entire
    chain (decoded to RGB) at native resolution, with no interpolation.
    """
    n_lr = len(lr_vals)
    n_k = len(k_vals)
    L = final_curves[0][2].shape[0]          # number of images in the chain
    # Get image size from a dummy decode
    dummy = Phi_fn(final_curves[0][2][:1])   # [1, C, H, W]
    _, C, H, W = dummy.shape                # e.g., (1, 3, 64, 64)
    strip_width = L * W
    strip_height = H

    # Figure dimensions (in inches) exactly match the total pixel grid
    fig_width = n_k * strip_width / dpi
    fig_height = n_lr * strip_height / dpi

    fig, axes = plt.subplots(n_lr, n_k, figsize=(fig_width, fig_height), dpi=dpi)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    # Ensure axes is always 2D
    if n_lr == 1 and n_k == 1:
        axes = np.array([[axes]])
    elif n_lr == 1:
        axes = axes[None, :]
    elif n_k == 1:
        axes = axes[:, None]

    for idx, (lr, k, curve) in enumerate(final_curves):
        row = idx // n_k
        col = idx % n_k
        ax = axes[row, col]

        # Decode whole chain and build a single strip tensor [C, H, strip_width]
        imgs = Phi_fn(curve)                      # [L, C, H, W]
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)     # back to [0,1]
        strip = torch.cat([imgs[i] for i in range(L)], dim=2)  # concat along width
        strip_np = strip.cpu().permute(1, 2, 0).numpy()

        ax.imshow(strip_np, aspect='equal', interpolation='none')
        ax.axis('off')

        # Add parameter text at top-left corner (tiny font, black on white border)
        ax.text(0.01, 0.99, f"lr={lr}, k={k}",
                transform=ax.transAxes, fontsize=4,
                verticalalignment='top', color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))

    # Save without additional padding or resizing
    plt.savefig(save_path, dpi=dpi, pad_inches=0, bbox_inches=None)
    plt.close()
    print(f"Saved full-resolution sweep summary to {save_path}")

# ------------------------
# Main
# ------------------------
@torch.no_grad()
def main():
    device = "cuda"  # or "cpu"
    image_path_A = "./andrewmvd/animal-faces/versions/1/afhq/train/dog/flickr_dog_000070.jpg" # Replace with your first training point filepath
    image_path_B = "./andrewmvd/animal-faces/versions/1/afhq/train/cat/flickr_cat_000070.jpg" # Replace with your second training point filepath

    x0_A = load_image(image_path_A).to(device)
    x0_B = load_image(image_path_B).to(device)

    model = UNetSD().to(device)
    scheduler = VPScheduler(num_timesteps=1000)

    checkpoint_path = ",/vp_diffusion_outputs/unet_animal_epoch_2000.pt" # Replace with your model checkpoint
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded trained model.")
    else:
        print("Checkpoint not found. Using random weights.")
    model.eval()

    print("Inverting Image A (Dog)...")
    xA_ = ddim_invert(model, scheduler, x0_A, num_steps=1000)
    print("Inverting Image B (Cat)...")
    xB_ = ddim_invert(model, scheduler, x0_B, num_steps=1000)

    @torch.no_grad()
    def Phi_fn(z):
        return Phi(z, model, scheduler, num_steps=1000)

    t_idx = 400
    n_geo_steps = 10
    n_iters = 100

    # Initial linear curve (used as starting point for all sweeps)
    curve = []
    for i in range(n_geo_steps):
        s = i / (n_geo_steps - 1)
        y_s = (1 - s) * xA_ + s * xB_
        curve.append(y_s)
    curve = torch.cat(curve, dim=0)  # [L, C, H, W]

    lr_values = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    k_values = [0.0, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]

    final_curves_list = []   # will store (lr, k, final_curve_tensor)

    for lr in lr_values:
        for k in k_values:
            print(f"\n--- Running NEB: lr={lr}, k={k} ---")
            curves_refined = neb_descent_on_score(
                curve.clone(),
                model=model,
                scheduler=scheduler,
                t_idx=t_idx,
                n_iters=n_iters,
                lr=lr,
                k=k,
                reparam_every=5
            )
            final_curve = curves_refined[-1]   # last refined chain
            final_curves_list.append((lr, k, final_curve))

    # Create single summary figure
    plot_sweep_summary(
        final_curves_list,
        lr_values,
        k_values,
        Phi_fn,
        save_path="NEB_sweep_summary.png"
    )

if __name__ == "__main__":
    main()
