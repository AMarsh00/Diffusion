"""
High_Confidence_Curves.py
Alexander Marsh
7 March 2026

Computes a high-confidence curve interpolation between two random generations from the CelebA-HQ model.
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
# DDIM deterministic sampling
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
# Path-Integral Metric & Levi-Civita
# ------------------------
def path_integral_metric(x, model, scheduler, t_idx, x_ind=None, beta=1.0, n_steps_int=10):
    """
    Compute G = I + exp(beta * integral) * (score score^T)
    x: [dim] tensor
    x_ind: reference point for integral (default origin)
    n_steps_int: number of steps to discretize path integral
    """
    if x_ind is None:
        x_ind = torch.zeros_like(x)

    dx = (x - x_ind) / n_steps_int
    integral = 0.0
    xk = x_ind.clone().detach()

    for _ in range(n_steps_int):
        s = score_fn(xk, model, scheduler, t_idx)  # score at xk
        integral += torch.dot(s.view(-1), dx.view(-1))
        xk = xk + dx

    s_x = score_fn(x, model, scheduler, t_idx).view(-1, 1)
    G = torch.eye(len(x), device=x.device) + torch.exp(beta * integral) * (s_x @ s_x.T)
    return G

def levi_civita_exp_map(x, v, model, scheduler, t_idx, beta=1.0, x_ind=None, n_steps=10, n_steps_int=10):
    """
    Levi-Civita exponential map using path-integral metric with Sherman-Morrison
    """
    x_curr = x.clone().detach()
    v_step = v / n_steps

    for _ in range(n_steps):
        # Compute metric (rank-1)
        if x_ind is None:
            x_ind = torch.zeros_like(x_curr)

        #dx_int = (x_curr - x_ind) / n_steps_int
        #integral = 0.0
        #xk = x_ind.clone().detach()
        #for _ in range(n_steps_int):
        #    s = score_fn(xk, model, scheduler, t_idx).view(-1)
        #    integral += torch.dot(s, dx_int.view(-1))
        #    xk = xk + dx_int

        s_x = score_fn(x_curr, model, scheduler, t_idx).view(-1)
        w = beta#torch.exp(beta * integral)

        # Solve G dx = v_step via Sherman-Morrison
        s_dot_s = (s_x**2).sum()
        s_dot_v = (s_x * v_step.view(-1)).sum()
        dx = v_step.view(-1) - w * s_x * s_dot_v / (1 + w * s_dot_s)
        dx = dx.view_as(x_curr)

        x_curr = x_curr + dx

    return x_curr.detach()

# ------------------------
# Log map shooting
# ------------------------
@torch.no_grad()
def log_map_shooting(y, y_target, model, scheduler, t_idx,
                     beta=1e6, max_iters=1000, lr=0.1,
                     n_substeps_schedule=[1, 2, 4, 8],):
    y = y.detach()
    y_target = y_target.detach()
    tol = 1e-2
    momentum_gamma = 0.9

    v = (y_target - y).detach().clone()
    v.requires_grad_(True)
    momentum = torch.zeros_like(v)
    best_v = v.clone()
    best_loss = float('inf')

    for idx, n_substeps in enumerate(n_substeps_schedule):
        n_iters = max_iters // len(n_substeps_schedule)
        for i in range(n_iters):
            y_pred = levi_civita_exp_map(y, v, model, scheduler, t_idx,
                                         beta=beta, n_steps=n_substeps)
            residual = y_target - y_pred
            loss = residual.norm()**2

            if loss.item() < tol:
                break

            step = lr * residual / (residual.norm() + 1e-8)
            momentum = momentum_gamma * momentum + step
            v = v + momentum

            if n_substeps == n_substeps_schedule[-1] and loss.item() < best_loss:
                best_loss = loss.item()
                best_v = v.clone()

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

@torch.no_grad()
def Phi(y, model, scheduler, num_steps=100, eta=0.0):
    timesteps = torch.linspace(scheduler.num_timesteps-1, 0, num_steps, dtype=torch.long)
    x = ddim_sample(model, scheduler, y, timesteps, eta=eta)
    return x

# ------------------------
# Dataset
# ------------------------
class CelebAHQDataset(Dataset):
    def __init__(self, root_dir, image_size=64):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)
        
@torch.no_grad()
def expected_primitive(
    z0,
    Z,
    Phi_fn,
    log_map_fn,
    exp_map_fn,
    eps,
    max_iter=200,
    lr=0.01
):
    """
    Compute the expected primitive of z0 w.r.t. a set of candidates Z
    using a diffusion model Phi_fn and Riemannian log map.
    """
    z_t = z0.clone()

    # ---- Cache Phi(Z) once ----
    Phi_Z = Phi_fn(Z)                       # [N,C,H,W]
    Zi_norm2 = (Z**2).flatten(1).sum(dim=1) # [N]

    for _ in range(max_iter):
        Phi_zt = Phi_fn(z_t)                # [1,C,H,W]
        zt_norm2 = (z_t**2).sum()

        # ---- Broadcast Phi_zt for batch log map ----
        Phi_zt_rep = Phi_zt.repeat(Z.shape[0], 1, 1, 1)  # [N,C,H,W]

        logs = log_map_fn(Phi_zt_rep, Phi_Z)            # [N,C,H,W]

        # ---- Compute weights and gradient ----
        log_w = 0.5 * (zt_norm2 - Zi_norm2)
        log_w = log_w - log_w.max()          # stabilize
        weights = torch.exp(log_w)
        weights = weights / (weights.sum() + 1e-8)
        weights = weights.view(-1,1,1,1)
        grad = -2 * (weights * logs).sum(dim=0, keepdim=True)           # [1,C,H,W]

        grad_norm = grad.norm() + 1e-8
        grad = grad / grad_norm

        # ---- Update z_t ----
        z_t = z_t - lr * grad

    return z_t

@torch.no_grad()
def refine_high_confidence_curve(
    curve,
    Phi_fn,
    log_map_fn,
    exp_map_fn,
    eps=0.5,
    n_iters=5,
    n_candidates=8,
    conf_kwargs={}
):
    """
    Refine a latent-space curve to maximize confidence under the diffusion model.
    curve: [L+1, C, H, W] NOT STRING
    """
    curves = [curve.clone()]

    for it in range(n_iters):
        print(f"\nRefinement iteration {it}")
        new_curve = curve.clone()

        for i in range(1, curve.shape[0] - 1):
            z0 = curve[i:i+1]

            # ---- Sample local latent neighborhood ----
            candidates = z0 + eps * torch.randn(
                n_candidates,
                *z0.shape[1:],
                device=z0.device
            )
            
            # ---- Compute expected primitive ----
            new_curve[i] = expected_primitive(
                z0,
                candidates,
                Phi_fn,
                log_map_fn,
                exp_map_fn,
                eps,
                max_iter=conf_kwargs.get("max_iter", 200),
                lr=conf_kwargs.get("lr", 0.01)
            )
        print(torch.norm(new_curve-curve))
        curve = new_curve
        curves.append(curve.clone())

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
    plt.savefig("HC_Curves.png")

# ------------------------
# Main
# ------------------------
@torch.no_grad()
def main():
    device = "cuda"#"cpu"  # or "cuda"
    image_path_A = "/data5/accounts/marsh/Diffusion/celeba_hq_prepared/000100.png"

    x0_A = load_image(image_path_A).to(device)
    xA_ = torch.randn_like(x0_A)
    xB_ = torch.randn_like(x0_A)

    model = UNetSD().to(device)
    scheduler = VPScheduler(num_timesteps=1000)

    checkpoint_path = "/data5/accounts/marsh/Diffusion/vp_diffusion_outputs/unet_epoch_2000.pt"
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded trained model.")
    else:
        print("Checkpoint not found. Using random weights.")
    model.eval()

    xA = Phi(xA_, model, scheduler, num_steps=100)
    xB = Phi(xB_, model, scheduler, num_steps=100)

    t_idx = 400
    beta = 10000.0
    n_geo_steps = 10
    lam = 1e6

    xA2 = ddim_sample(model, scheduler, xA_, torch.linspace(999-t_idx, 0, 1000-t_idx, dtype=torch.long, device=device))
    xB2 = ddim_sample(model, scheduler, xB_, torch.linspace(999-t_idx, 0, 1000-t_idx, dtype=torch.long, device=device))

    @torch.no_grad()
    def Phi_fn(z):
        return Phi(z, model, scheduler, num_steps=1000)

    def log_map_fn(x, y):
        return log_map_shooting(
            y=x,
            y_target=y,
            model=model,
            scheduler=scheduler,
            t_idx=t_idx,
            beta=beta,
            max_iters=50
        )

    def exp_map_fn(x, v):
        return exp_map(
            x,
            v,
            model=model,
            scheduler=scheduler,
            t_idx=t_idx,
            lam=lam,
            beta=beta
        )

    # Initial linear curve
    curve = []
    for i in range(n_geo_steps):
        s = i / (n_geo_steps - 1)
        y_s = (1 - s) * xA_ + s * xB_
        curve.append(y_s)
    curve = torch.cat(curve, dim=0)  # [L, C, H, W]

    # -------------------------
    # Refine curve and plot after each iteration
    # -------------------------
    curves_refined = refine_high_confidence_curve(
        curve,
        Phi_fn=Phi_fn,
        log_map_fn=log_map_fn,
        exp_map_fn=exp_map_fn,
        eps=1.0,
        n_iters=5,
        n_candidates=16,
        conf_kwargs=dict(max_iter=20, lr=0.005)
    )
    
    # -------------------------
    # Plot all iterations at once
    # -------------------------
    plot_curve_images(
        curves_refined,
        Phi_fn,
        title="High-Confidence Curve Refinement"
    )
    print("Saved high-confidence curve refinement figure with all iterations.")

if __name__ == "__main__":
    main()
        
