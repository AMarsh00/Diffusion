"""
Geodesics.py
Alexander Marsh
27 January 2026

Computes geodesics between random generations of the CelebA-HQ diffusion model with our metric for various values of theta.
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

# ------------------------
# UNetSD Components
# ------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timestep):
        device = "cpu"#timestep.device
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

        dx_int = (x_curr - x_ind) / n_steps_int
        integral = 0.0
        xk = x_ind.clone().detach()
        for _ in range(n_steps_int):
            s = score_fn(xk, model, scheduler, t_idx).view(-1)
            integral += torch.dot(s, dx_int.view(-1))
            xk = xk + dx_int

        s_x = score_fn(x_curr, model, scheduler, t_idx).view(-1)
        w = torch.exp(beta * integral)

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
def log_map_shooting(y, y_target, model, scheduler, t_idx, beta=1e6, max_iters=1000, lr=0.1, n_substeps_schedule=[1, 2, 4, 8],):
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
def Phi(y, model, scheduler, num_steps=50, eta=0.0):
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

# ------------------------
# Main
# ------------------------
def main():
    device = "cpu"  # or "cuda" if available
    image_path_A = "/data5/accounts/marsh/Diffusion/celeba_hq_prepared/000100.png" # Replace with your filepath

    x0_A = load_image(image_path_A).to(device)
    xA_ = torch.randn_like(x0_A)
    xB_ = torch.randn_like(x0_A)

    model = UNetSD().to(device)
    scheduler = VPScheduler(num_timesteps=1000)

    checkpoint_path = "/data5/accounts/marsh/Diffusion/vp_diffusion_outputs/unet_epoch_2000.pt" # Replace with your filepath
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded trained model.")
    else:
        print("Checkpoint not found. Using random weights.")
    model.eval()

    xA = Phi(xA_, model, scheduler, num_steps=1000)
    xB = Phi(xB_, model, scheduler, num_steps=1000)

    # Dataset
    train_dataset = CelebAHQDataset(root_dir="/data5/accounts/marsh/Diffusion/celeba_hq_prepared", image_size=64) # Replace with your filepath
    loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    train_images = torch.cat([b for b in loader], dim=0)

    # Timestep for metric
    t_idx = 400
    beta_values = [-50.0, -25.0, -10.0, -5.0, -2.5, -1.0, 0]
    n_geo_steps = 10
    all_geodesics = {}
    lam = 1e6

    # Noisy starting points
    xA2 = ddim_sample(model, scheduler, xA_, torch.linspace(999-t_idx, 0, 1000-t_idx, dtype=torch.long, device=device))
    xB2 = ddim_sample(model, scheduler, xB_, torch.linspace(999-t_idx, 0, 1000-t_idx, dtype=torch.long, device=device))

    for beta in beta_values:
        print(f"Computing bidirectional geodesic for beta={beta}...")
        v_forward = log_map_shooting(xA2, xB2, model, scheduler, t_idx, beta=beta)
        v_backward = log_map_shooting(xB2, xA2, model, scheduler, t_idx, beta=beta)

        geodesic = []
        for i in range(n_geo_steps):
            s = i / (n_geo_steps - 1)
            if s <= 0.5:
                y_s = levi_civita_exp_map(xA2, s * v_forward, model, scheduler, t_idx, beta=beta)
            else:
                y_s = levi_civita_exp_map(xB2, (1-s) * v_backward, model, scheduler, t_idx, beta=beta)

            x0_pred = ddim_sample(model, scheduler, y_s,
                                  torch.linspace(t_idx-1, 0, t_idx, dtype=torch.long, device=device))
            geodesic.append(x0_pred.detach().cpu())

        all_geodesics[beta] = geodesic
        print(f"Geodesic complete for beta={beta}.")

    # Linear interpolation
    linear_interp = []
    for i in range(n_geo_steps):
        s = i / (n_geo_steps - 1)
        y_s = (1 - s) * xA2 + s * xB2
        x_s = ddim_sample(model, scheduler, y_s,
                          torch.linspace(t_idx-1, 0, t_idx, dtype=torch.long, device=device))
        linear_interp.append(x_s.detach().cpu())

    # Plot results
    os.makedirs("geodesic_frames", exist_ok=True)
    width, height = (64, 64)
    label_height = 15
    total_width = width * n_geo_steps
    total_height = height * len(beta_values) + height + label_height * (len(beta_values) + 1)
    final_image = Image.new("RGB", (total_width, total_height), color=(0,0,0))
    try:
        font = ImageFont.load_default()
    except:
        font = None

    for row_idx, beta in enumerate(beta_values):
        geodesic = all_geodesics[beta]
        y_offset = row_idx * (height + label_height)
        if font:
            draw = ImageDraw.Draw(final_image)
            draw.text((0, y_offset), f"theta={beta}", fill=(255,0,0), font=font)
        y_offset += label_height
        for col_idx, frame in enumerate(geodesic):
            img = frame.squeeze(0)
            img = (img * 0.5 + 0.5).clamp(0, 1)
            pil_img = transforms.ToPILImage()(img.cpu())
            final_image.paste(pil_img, (col_idx * width, y_offset))

    # Linear interpolation row
    y_offset = len(beta_values) * (height + label_height)
    if font:
        draw = ImageDraw.Draw(final_image)
        draw.text((0, y_offset), "LERP", fill=(255,0,0), font=font)
    y_offset += label_height
    for col_idx, frame in enumerate(linear_interp):
        img = frame.squeeze(0)
        img = (img * 0.5 + 0.5).clamp(0, 1)
        pil_img = transforms.ToPILImage()(img.cpu())
        final_image.paste(pil_img, (col_idx * width, y_offset))

    final_image.save("geodesic_vs_lerp.png")
    print("Saved geodesic_vs_lerp.png")

if __name__ == "__main__":
    main()
