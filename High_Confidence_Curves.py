"""
High_Confidence_Curves.py
Alexander Marsh
2 January 2026

Computes a high-confidence curve interpolation between two random generations from the CelebA-HQ model.
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

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
#@torch.no_grad()
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
    """
    Computes the Stein score s_theta(x, t) = grad_x log p_t(x)
    using epsilon model output.
    x: [B, C, H, W]
    t: int or tensor of shape [B]
    """
    device = x.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    if isinstance(t, int):
        t_tensor = torch.tensor([t], device=device)
        alpha_bar = alphas_cumprod[t]
    else:
        t_tensor = t
        alpha_bar = alphas_cumprod[t]

    epsilon_theta = model(x, t_tensor)  # [B,C,H,W]
    score = - epsilon_theta / torch.sqrt(1 - alpha_bar)  # [B,C,H,W]
    return score

# ------------------------
# Log map on image space (Euclidean approx)
# ------------------------
# ------------------------
# Linear path integral
# ------------------------
def find_nearest_training_point(y, train_images):
    """
    y: [B, C, H, W]
    train_images: [N, C, H, W]
    returns: [B, C, H, W] nearest image for each y in batch
    """
    B = y.shape[0]
    nearest = []

    # Flatten images for distance computation
    y_flat = y.view(B, -1)                # [B, C*H*W]
    train_flat = train_images.view(len(train_images), -1)  # [N, C*H*W]

    for i in range(B):
        dists = ((train_flat - y_flat[i:i+1])**2).sum(dim=1)  # [N]
        idx = dists.argmin()
        nearest.append(train_images[idx])

    return torch.stack(nearest, dim=0)  # [B, C, H, W]
    
@torch.no_grad()
def linear_path_integral_fast(y, y_target, model, scheduler, t_idx, n_steps=16):
    # y, y_target: [B, C, H, W]

    B = y.shape[0]
    phi = torch.zeros(B, device=y.device)

    # Flatten everything except batch dim
    y_flat = y.flatten(1)                # [B, D]
    y_target_flat = y_target.flatten(1)  # [B, D]

    direction = y_target_flat - y_flat   # [B, D]
    ds = 1.0 / (n_steps - 1)

    for i in range(n_steps):
        s = i * ds

        # Interpolate in flattened form
        y_s_flat = y_flat + s * direction       # [B, D]
        y_s = y_s_flat.view_as(y)               # back to [B, C, H, W]

        # Score function still expects 4D input
        score = score_fn(y_s, model, scheduler, t_idx)  # [B, C, H, W]
        score_flat = score.flatten(1)                   # [B, D]

        # inner product <score, direction>
        inner = (score_flat * direction).sum(dim=1)     # [B]

        phi += inner * ds

    return phi

# ------------------------
# Exponential map with new metric
# ------------------------
@torch.no_grad()
def exp_map(y, v, model, scheduler, t_idx, train_images=None,
                      lam=1000.0, beta=0.0, n_substeps=8, correct_every=1):
    """
    Exponential map along tangent vector `v` with score correction using the diffusion model.

    Parameters:
    - y: [B,C,H,W] starting point
    - v: [B,C,H,W] tangent vector
    - model: trained UNetSD
    - scheduler: VPScheduler
    - t_idx: diffusion timestep
    - train_images: optional dataset for nearest neighbor
    - lam, beta: Riemannian metric parameters
    - n_substeps: number of substeps
    - correct_every: how often to apply diffusion-based correction
    """
    y_current = y
    dx = v / n_substeps

    for step in range(n_substeps):
        # Compute score
        s = score_fn(y_current, model, scheduler, t_idx)  # [B,C,H,W]

        if beta != 0:
            # Optional: nearest training point
            if train_images is not None:
                y_target = find_nearest_training_point(y_current, train_images)
            else:
                y_target = torch.zeros_like(y_current)
    
            # Path integral factor
            Phi_val = linear_path_integral_fast(y_current, y_target, model, scheduler, t_idx)  # [B]
            #print(Phi_val)
            #Phi_val = torch.clamp(Phi_val, -1, 1)
            C = torch.exp(beta * Phi_val).view(-1, 1, 1, 1)  # broadcast
        else:
            C = 1

        # Riemannian metric step
        s_norm2 = (s * s).view(s.shape[0], -1).sum(dim=1, keepdim=True)  # [B,1]
        proj = (s * dx).view(s.shape[0], -1).sum(dim=1, keepdim=True)    # [B,1]
        denom = 1 + C * s_norm2
        ginv_dx = dx - C * (proj / denom).view(-1,1,1,1) * s# * (dx.norm() / (s * s).norm())
        y_current = y_current + ginv_dx#C * ginv_dx

        """# Diffusion-based correction (optional)
        if (step + 1) % correct_every == 0:
            t_tensor = torch.tensor([t_idx], device=y_current.device)
            eps = model(y_current, t_tensor)
            alpha_bar = scheduler.alphas_cumprod[t_idx]
            x0_pred = (y_current - eps * torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_bar)
            y_current = torch.sqrt(alpha_bar) * x0_pred + eps * torch.sqrt(1 - alpha_bar)"""
    """if torch.norm(v) > 0.1:
        displacement = y_current - y
        scale = torch.norm(v) / torch.norm(displacement)
        y_current = y + scale * displacement"""
    return y_current
    
# ------------------------
# Log map shooting with new metric
# ------------------------
@torch.no_grad()
def log_map_shooting(y, y_target, model, scheduler, t_idx,
                     lam=1000.0, beta=0.0, max_iters=1000, lr=0.1,
                     n_substeps_schedule=[1,2,4,8], train_images=None):
    z=torch.norm(y-y_target)
    y = y.detach().clone()
    y_target = y_target.detach().clone()
    tol = 1e-1
    momentum_gamma = 0.9
    #print(torch.norm(y-y_target))
    
    v = (y_target - y).detach().clone()
    v.requires_grad_(True)
    #print(torch.norm(v))
    
    momentum = torch.zeros_like(v)
    best_v = v.clone()
    best_loss = float('inf')

    for idx, n_substeps in enumerate(n_substeps_schedule):
        n_iters = max_iters // len(n_substeps_schedule)
        V = best_v
        #print(torch.norm(y-y_target))
        for i in range(n_iters):
            #print(torch.norm(y-y_target))
            y_pred = exp_map(y.detach().clone(), v.detach().clone(), model, scheduler, t_idx,
                             lam=lam, beta=beta, n_substeps=n_substeps,
                             train_images=train_images)
            residual = y_target - y_pred

            loss = residual.norm()**2
            #print(torch.norm(y-y_target))
            #print(z)
            #print(f"n_substeps={n_substeps}, iter={i}, loss={loss.item():.6f}")
            
            if loss.item() < tol:
                #print('b')
                break
            
            #quick_exit(0)
            # normalize residual to prevent huge jumps
            step = lr * residual / (residual.norm() + 1e-8)

            # apply momentum
            momentum = momentum_gamma * momentum + step
            v = v + momentum

            # track best v only at the **max substeps stage**
            if n_substeps == n_substeps_schedule[-1] and loss.item() < best_loss:
                best_loss = loss.item()
                best_v = v.clone()

    return best_v

    """# detach inputs
    y = y.detach()
    y_target = y_target.detach()

    # initial tangent guess
    v = 0*(y_target - y).detach().clone().requires_grad_(True)

    for it in range(max_iters):

        # ---- 1. Forward: compute predicted endpoint ----
        y_pred = exp_map(
            y, v,
            model=model, scheduler=scheduler,
            t_idx=t_idx,
            lam=lam, beta=beta,
            n_substeps=n_substeps
        )

        # ---- 2. Compute loss ----
        residual = y_pred - y_target
        loss = 0.5 * residual.pow(2).sum()

        print(f"[adjoint] iter={it}   loss={loss.item():.6f}")

        if loss.item() < 1e-5:
            break

        # ---- 3. Backprop to get adjoint gradient: J^T r ----
        loss.backward()

        # ---- 4. Gradient update ----
        with torch.no_grad():
            grad = v.grad.clone()
            v -= lr * grad

            # Optional: damp explosion
            if v.norm() > 5.0:
                v *= 5.0 / v.norm()

        v.grad.zero_()

    return v.detach()"""

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

#@torch.no_grad()
def Phi(y, model, scheduler, num_steps=50, eta=0.0, t_idx=0):
    """
    Map noise y -> image x using DDIM deterministic sampling.
    y: input noise tensor [B,C,H,W]
    """
    # Define timesteps for DDIM
    timesteps = torch.linspace(scheduler.num_timesteps-t_idx-1, 0, scheduler.num_timesteps-t_idx, dtype=torch.long)
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
    z0, Z, Phi_fn, log_map_fn, exp_map_fn, eps,
    max_iter=200, lr=0.01
):
    z_t = z0.clone()

    # 🔥 CACHE Phi(Z) ONCE
    Phi_Z = Phi_fn(Z)   # [N, C, H, W]
    Zi_norm2 = (Z**2).view(Z.shape[0], -1).sum(dim=1)

    for it in range(max_iter):
        grad = torch.zeros_like(z_t)

        Phi_zt = Phi_fn(z_t)
        zt_norm2 = (z_t**2).sum()

        for i in range(Z.shape[0]):
            logs = log_map_fn(Phi_zt, Phi_Z[i:i+1])
            w = torch.exp(0.5 * (zt_norm2 - Zi_norm2[i]))
            grad -= 2 * w * logs

        z_next = z_t - lr * grad

        if (z_next - z0).view(-1).norm() >= eps:
            break

        z_t = z_next

    return z_t

def confidence_metric(
    z: torch.Tensor,            # latent space point x
    Phi_fn,                      # diffusion model: latent -> image space
    log_map_fn,                  # logarithmic map for geodesic distance
    exp_map_fn,                  # exponential map
    eps: float = 1.0,            # radius for the e-ball
    N: int = 400,                # Monte Carlo samples
    max_iter: int = 200,         # max iterations for expected primitive
    lr: float = 0.01             # learning rate for expected primitive
):
    device = z.device

    # -------------------------
    # 1. Construct e-ball in latent space and map to image space
    # -------------------------
    def sample_ball(z0, eps, N):
        shape = z0.shape
        d = z0.numel() // z0.shape[0]  # flattened dimension per sample
        # Generate random directions
        u = torch.randn(N, *shape[1:], device=z0.device)
        u = u.view(N, -1)
        # Normalize each vector to have norm 1
        u = u / u.norm(dim=1, keepdim=True)
        # Scale by eps
        u = eps * u
        # Reshape back and add to z0
        return z0 + u.view(N, *shape[1:])

    Z = sample_ball(z, eps, N)# FUNDAMENTAL BUG
    
    # -------------------------
    # 2. Map samples to region R
    # -------------------------
    #Y = Phi_fn(Z)   # [N,C,H,W] or [N,C,...]
    
    # -------------------------
    # 3. Compute expected primitive
    # -------------------------
    y_star = expected_primitive(
        z0=z,
        Z=Z,
        Phi_fn=Phi_fn,
        log_map_fn=log_map_fn,
        exp_map_fn=exp_map_fn,
        eps=eps,
        max_iter=max_iter,
        lr=lr
    )

    # -------------------------
    # 4. Compute geodesic distances
    # -------------------------
    geodesic_dists = []
    for i in range(N):
        log_i = log_map_fn(y_star, Z[i:i+1])
        dist_i = (log_i**2).sum().item()
        geodesic_dists.append(dist_i)
    geodesic_dists = torch.tensor(geodesic_dists, device=device)

    # Distance from center to expected primitive
    log_center = log_map_fn(Phi_fn(y_star), Phi_fn(z))
    dist_center = (log_center**2).sum().item()

    # -------------------------
    # 5. Compute weighted Riemannian variance
    # -------------------------
    z_star_norm = (y_star**2).sum()
    Z_norm = (Z**2).view(N, -1).sum(dim=1)
    weights = torch.exp(0.5 * (z_star_norm - Z_norm))
    var_R = (weights * geodesic_dists).mean().item()

    # -------------------------
    # 6. Compute confidence metric
    # -------------------------
    C = torch.log(torch.tensor(var_R)) + dist_center**2 / (var_R+1e-5)

    return C.item(), var_R, dist_center

import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

# ============================================================
# Geodesic curve in latent space (USING YOUR LOG/EXP MAP)
# ============================================================
@torch.no_grad()
def geodesic_curve_latent(
    z0,
    z1,
    L,
    log_map_fn,
    exp_map_fn
):
    """
    z0, z1: [1,C,H,W]
    returns: [L+1,C,H,W]
    """
    v = log_map_fn(z0, z1)
    curve = []

    ts = torch.linspace(0, 1, L + 1, device=z0.device)
    for t in ts:
        curve.append(exp_map_fn(z0, t * v))

    return torch.cat(curve, dim=0)


# ============================================================
# High-confidence curve refinement (MAIN ALGORITHM)
# ============================================================
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
    curve: [L+1,C,H,W]
    """
    curves = [curve.clone()]

    for it in range(n_iters):
        print(f"\nRefinement iteration {it}")
        new_curve = curve.clone()

        for i in range(1, curve.shape[0] - 1):
            z0 = curve[i:i+1]

            # local latent neighborhood
            candidates = z0 + eps * torch.randn(
                n_candidates,
                *z0.shape[1:],
                device=z0.device
            )

            costs = torch.zeros(n_candidates, device=z0.device)
            
            for j in range(n_candidates):
                costs[j], _, _ = confidence_metric(
                    candidates[j:j+1],
                    Phi_fn=Phi_fn,
                    log_map_fn=log_map_fn,
                    exp_map_fn=exp_map_fn,
                    **conf_kwargs
                )

            costs = torch.tensor(costs)
            best_idx = costs.argmin()
            new_curve[i] = candidates[best_idx]

            print(f"  point {i}: best C = {costs[best_idx]:.4f}")

        curve = new_curve
        curves.append(curve.clone())

    return curves


# ============================================================
# Visualization helper
# ============================================================
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
    #plt.show()

# ============================================================
# MAIN
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"#"cpu"

    # -------------------------
    # Load model & scheduler
    # -------------------------
    model = UNetSD().to(device)
    scheduler = VPScheduler(num_timesteps=1000)

    checkpoint_path = "/data5/accounts/marsh/Diffusion/vp_diffusion_outputs/unet_epoch_2000.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()

    # -------------------------
    # Diffusion settings
    # -------------------------
    t_idx = 400
    lam = 0.0
    beta = -10.0

    ddim_timesteps = torch.linspace(
        scheduler.num_timesteps - t_idx - 1,
        0,
        scheduler.num_timesteps - t_idx,
        dtype=torch.long,
        device=device
    )
    
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    @torch.no_grad()
    def Phi_fn(z):
        return ddim_sample(
            model,
            scheduler,
            z,
            ddim_timesteps
        )

    def log_map_fn(x, y):
        return log_map_shooting(
            y=x,
            y_target=y,
            model=model,
            scheduler=scheduler,
            t_idx=t_idx,
            lam=lam,
            beta=beta,
            max_iters=200,
        )

    def exp_map_fn(x, v):
        return exp_map(
            x,
            v,
            model=model,
            scheduler=scheduler,
            t_idx=t_idx,
            lam=lam,
            beta=beta,
        )

    # -------------------------
    # Endpoints
    # -------------------------
    z_start = torch.randn(1, 3, 64, 64, device=device)
    z_end   = z_start + 0.7 * torch.randn_like(z_start)

    # -------------------------
    # Initial geodesic
    # -------------------------
    L = 5
    init_curve = geodesic_curve_latent(
        z_start,
        z_end,
        L=L,
        log_map_fn=log_map_fn,
        exp_map_fn=exp_map_fn
    )

    # -------------------------
    # High-confidence refinement
    # -------------------------
    curves = refine_high_confidence_curve(
        init_curve,
        Phi_fn=Phi_fn,
        log_map_fn=log_map_fn,
        exp_map_fn=exp_map_fn,
        eps=1.0,
        n_iters=3,
        n_candidates=3,
        conf_kwargs=dict(
            eps=1.0,
            N=16,
            max_iter=2,
            lr=0.01
        )
    )

    # -------------------------
    # Visualize
    # -------------------------
    plot_curve_images(curves, Phi_fn,
        title="High-Confidence Curve Refinement (Diffusion)"
    )


if __name__ == "__main__":
    main()
