"""
Confidence.py
Alexander Marsh
27 January 2026

Computes and displays confidence scores for 16 random generations
"""

import os
import math
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

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
# DDIM deterministic sampling
# ------------------------
@torch.no_grad()
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

# ------------------------
# Stein Score / Riemannian metric
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
# Levi-Civita Exponential Map
# ------------------------
@torch.no_grad()
def levi_civita_exp_map(y, v, model, scheduler, t_idx, beta=1.0, n_substeps=8):
    y_curr = y.clone()
    dx = v / n_substeps
    for _ in range(n_substeps):
        s = score_fn(y_curr, model, scheduler, t_idx)
        s_flat = s.flatten(1)
        dx_flat = dx.flatten(1)
        s_dot_dx = (s_flat * dx_flat).sum(dim=1, keepdim=True)
        s_norm2 = (s_flat * s_flat).sum(dim=1, keepdim=True)
        factor = beta * s_dot_dx / (1 + beta * s_norm2)
        ginv_dx_flat = dx_flat - factor * s_flat
        ginv_dx = ginv_dx_flat.view_as(y_curr)
        y_curr = y_curr + ginv_dx
    return y_curr

# ------------------------
# Levi-Civita Log Map Shooting
# ------------------------
@torch.no_grad()
def levi_civita_log_map(Y, Y_target, model, scheduler, t_idx,
                              beta=1.0, n_substeps_schedule=[1,2,4,8],
                              max_iters=500, lr=0.1):
    """
    Vectorized Levi-Civita log map (batch version).

    Y: [N, C, H, W] starting points
    Y_target: [N, C, H, W] target points
    """
    N = Y.shape[0]

    # Detach inputs
    Y = Y.detach().clone()
    Y_target = Y_target.detach().clone()

    tol = 1e-3
    momentum_gamma = 0.9

    # Initial tangent vectors
    v = (Y_target - Y).clone().detach()
    v.requires_grad_(False)

    momentum = torch.zeros_like(v)
    best_v = v.clone()
    best_loss = torch.full((N,), float('inf'), device=Y.device)

    for n_substeps in n_substeps_schedule:
        n_iters = max_iters // len(n_substeps_schedule)
        for i in range(n_iters):
            # Compute predicted endpoints via exp map (vectorized)
            Y_pred = levi_civita_exp_map(Y, v, model, scheduler, t_idx,
                                         beta=beta, n_substeps=n_substeps)

            # Residuals and per-sample loss
            residual = Y_target - Y_pred          # [N, C, H, W]
            loss = residual.view(N, -1).norm(dim=1)**2  # [N]

            # Break if all below tolerance
            if (loss < tol).all():
                break

            # Step with normalization
            step = lr * residual / (residual.view(N, -1).norm(dim=1).view(N,1,1,1) + 1e-8)

            # Momentum update
            momentum = momentum_gamma * momentum + step
            v = v + momentum

            # Track best v for final substeps
            mask = (n_substeps == n_substeps_schedule[-1]) & (loss < best_loss)
            best_loss = torch.where(loss < best_loss, loss, best_loss)
            best_v = torch.where(mask.view(N,1,1,1), v, best_v)

    return best_v

# ------------------------
# Image helpers
# ------------------------
def load_image(path, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

# ------------------------
# Confidence Metric
# ------------------------
@torch.no_grad()
def confidence_metric(
    z: torch.Tensor,
    Phi_fn,
    log_map_fn,
    exp_map_fn,
    eps: float = 1.0,
    N: int = 16,
    max_iter: int = 200,
    lr: float = 0.01
):
    device = z.device
    shape = z.shape
    d = z.numel() // z.shape[0]

    # --- Sample N points on the epsilon-ball ---
    u = torch.randn(N, *shape[1:], device=device).view(N, -1)
    u = u / u.norm(dim=1, keepdim=True) * eps
    Z = z + u.view(N, *shape[1:])

    # --- Compute expected primitive (vectorized) ---
    z_t = z.clone()
    tol = 1e-3
    momentum_gamma = 0.9
    momentum = torch.zeros_like(z_t).repeat(N, *[1]*(len(z_t.shape)-1))
    for t in range(max_iter):
        # Expand z_t to match Z batch
        z_t_exp = z_t.expand(N, *z_t.shape[1:])
        Phi_zt = Phi_fn(z_t_exp)         # Shape: [N, C, H, W]
        Phi_Z = Phi_fn(Z)                # Shape: [N, C, H, W]
        logs = log_map_fn(Phi_zt, Phi_Z) # [N, C, H, W]
        
        # Compute weights: w = exp(0.5 * (||z_t||^2 - ||Z||^2))
        zt_norm2 = (z_t**2).sum()
        Z_norm2 = (Z**2).view(N, -1).sum(dim=1)
        w = torch.exp(0.5 * (zt_norm2 - Z_norm2)).view(N, 1, 1, 1)
        
        # Gradient step (vectorized)
        grad = -2 * (w * logs).mean(dim=0, keepdim=True)
        z_next = z_t - lr * grad

        if (z_next - z_t).view(-1).norm() < tol:
            break
        z_t = z_next

    y_star = z_t

    # --- Geodesic distances (vectorized) ---
    y_star_exp = y_star.expand(N, *y_star.shape[1:])
    log_vec = log_map_fn(y_star_exp, Z)
    geodesic_dists = (log_vec.view(N, -1)**2).sum(dim=1)

    log_center = log_map_fn(Phi_fn(y_star.unsqueeze(0)), Phi_fn(z))
    dist_center = (log_center**2).sum().item()

    z_star_norm = (y_star**2).sum()
    Z_norm = (Z**2).view(N, -1).sum(dim=1)
    weights = torch.exp(0.5 * (z_star_norm - Z_norm))
    var_R = (weights * geodesic_dists).mean().item()

    C = torch.log(torch.tensor(var_R)) + dist_center**2 / (var_R + 1e-5)
    return C.item(), var_R, dist_center

# ------------------------
# Main
# ------------------------
def main():
    device="cuda"#"cpu"
    t_idx = 400
    beta_values = [-50.0, -25.0, -10.0, -5.0, -2.5, -1.0, 0]
    n_geo_steps = 10
    all_geodesics = {}
    lam = 1e6
    num_samples = 16
    eps = 0.1
    beta = -5.0
    # Load model & scheduler
    model = UNetSD().to(device)
    scheduler = VPScheduler(num_timesteps=1000)
    checkpoint_path = "/data5/accounts/marsh/Diffusion/vp_diffusion_outputs/unet_epoch_2000.pt" # Replace with your path
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded trained model.")
    else:
        print("Checkpoint not found. Using random weights.")
    model.eval()

    # ------------------------
    # Timesteps for DDIM
    # ------------------------
    timesteps = torch.linspace(1000-t_idx-1, 0, 1000-t_idx, dtype=torch.long, device=device)

    # ------------------------
    # Phi_fn, log_map_fn, exp_map_fn
    # ------------------------
    Phi_fn = lambda z: ddim_sample(model, scheduler, z, timesteps, eta=0.0)
    log_map_fn = lambda x, y: levi_civita_log_map(
        Y=x, Y_target=y, model=model, scheduler=scheduler, t_idx=t_idx, beta=beta
    )
    exp_map_fn = lambda x, v: levi_civita_exp_map(
        y=x, v=v, model=model, scheduler=scheduler, t_idx=t_idx, beta=beta
    )

    # ------------------------
    # Generate batch of random z0
    # ------------------------
    z0 = torch.randn(num_samples, 3, 64, 64, device=device)  # [N, C, H, W]

    # ------------------------
    # Sample N points on epsilon-ball (for confidence)
    # ------------------------
    N = 16
    u = torch.randn(N, num_samples, 3, 64, 64, device=device)
    u = eps * u / (u.view(N, num_samples, -1).norm(dim=2).view(N, num_samples, 1, 1, 1) + 1e-8)
    Z = z0.unsqueeze(0) + u  # [N, num_samples, C, H, W]

    # Flatten for batch processing
    Z_flat = Z.view(N * num_samples, 3, 64, 64)

    # Precompute Phi(Z) for all N*num_samples points
    Phi_Z_flat = Phi_fn(Z_flat)
    Phi_Z = Phi_Z_flat.view(N, num_samples, 3, 64, 64)

    # ------------------------
    # Confidence metric vectorized
    # ------------------------
    max_iter = 2
    lr = 0.05
    tol = 1e-3

    z_t = z0.clone()
    for _ in range(max_iter):
        z_t_exp = z_t.unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, num_samples, C, H, W]
        z_t_flat = z_t_exp.reshape(N*num_samples, 3, 64, 64)
        Phi_zt_flat = Phi_fn(z_t_flat)
        Phi_zt = Phi_zt_flat.view(N, num_samples, 3, 64, 64)

        # Compute log maps
        logs = torch.zeros_like(Phi_zt)

        for i in range(num_samples):
            # Each call sees [N, C, H, W]
            logs[:, i] = log_map_fn(
                Phi_zt[:, i],   # [N, C, H, W]
                Phi_Z[:, i]     # [N, C, H, W]
            )

        # Weighting
        zt_norm2 = (z_t**2).sum()
        Z_norm2 = (Z_flat**2).view(N, num_samples, -1).sum(dim=2)
        w = torch.exp(0.5 * (zt_norm2 - Z_norm2)).view(N, num_samples, 1, 1, 1)

        # Gradient step (vectorized)
        grad = -2 * (w * logs).mean(dim=0)  # Average over N
        z_next = z_t - lr * grad

        if (z_next - z_t).view(-1).norm() < tol:
            break
        z_t = z_next

    y_star = z_t

    # Compute geodesic distances
    y_star_exp = y_star.unsqueeze(0).expand(N, -1, -1, -1, -1)
    log_vec = torch.zeros_like(Z)

    for i in range(num_samples):
        log_vec[:, i] = log_map_fn(
            y_star_exp[:, i],  # [N, C, H, W]
            Z[:, i]            # [N, C, H, W]
        )
    geodesic_dists = (log_vec.view(N, num_samples, -1)**2).sum(dim=2)

    dist_center = torch.zeros(num_samples, device=device)

    for i in range(num_samples):
        phi_y = Phi_fn(y_star[i:i+1])   # [1, C, H, W]
        phi_z = Phi_fn(z0[i:i+1])       # [1, C, H, W]
    
        log_c = log_map_fn(phi_y, phi_z)  # [1, C, H, W]
        dist_center[i] = (log_c**2).sum()
        
    dist_center = dist_center.cpu().numpy()

    z_star_norm = (y_star**2).sum()
    Z_norm = (Z**2).view(N, -1).sum(dim=1)
    weights = torch.exp(0.5 * (z_star_norm - Z_norm))
    var_R = (weights * geodesic_dists).mean().item()
    C = torch.log(torch.tensor(var_R)) + dist_center**2 / (var_R + 1e-5)

    # ------------------------
    # Generate final DDIM samples for visualization
    # ------------------------
    samples = ddim_sample(model, scheduler, z0, torch.linspace(999,0,1000,dtype=torch.long, device=device))
    samples = (samples * 0.5 + 0.5).clamp(0,1)

    # Plot 4x4 grid
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(12,12))
    for i in range(num_samples):
        ax = axes[i//4, i%4]
        img = transforms.ToPILImage()(samples[i])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Score: {C[i]:.4f}", fontsize=10)
    plt.tight_layout()
    os.makedirs("confidences", exist_ok=True)
    out_path = "confidences/epsilon_batch_samples.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved batch plot to {out_path}")

if __name__ == "__main__":
    main()

