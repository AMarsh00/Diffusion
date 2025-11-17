import os
import math
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont

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
            Phi_val = torch.clamp(Phi_val, -1, 1)
            C = torch.exp(beta * Phi_val).view(-1, 1, 1, 1)  # broadcast
        else:
            C = 1

        # Riemannian metric step
        s_norm2 = (s * s).view(s.shape[0], -1).sum(dim=1, keepdim=True)  # [B,1]
        proj = (s * dx).view(s.shape[0], -1).sum(dim=1, keepdim=True)    # [B,1]
        denom = 1 + lam * s_norm2
        ginv_dx = dx - lam * (proj / denom).view(-1,1,1,1) * s# * (dx.norm() / (s * s).norm())
        y_current = y_current + C * ginv_dx

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
                     n_substeps_schedule=[1, 2, 4, 8], train_images=None):
    y = y.detach()
    y_target = y_target.detach()
    tol = 1e-4
    momentum_gamma = 0.9
    
    v = (y_target - y).detach().clone()
    v.requires_grad_(True)
    
    momentum = torch.zeros_like(v)
    best_v = v.clone()
    best_loss = float('inf')

    for idx, n_substeps in enumerate(n_substeps_schedule):
        n_iters = max_iters // len(n_substeps_schedule)
        V = best_v
        for i in range(n_iters):
            y_pred = exp_map(y, v, model, scheduler, t_idx,
                             lam=lam, beta=beta, n_substeps=n_substeps,
                             train_images=train_images)
            residual = y_target - y_pred

            loss = residual.norm()**2
            print(f"n_substeps={n_substeps}, iter={i}, loss={loss.item():.6f}")

            if loss.item() < tol:
                break

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
    """
    Map noise y -> image x using DDIM deterministic sampling.
    y: input noise tensor [B,C,H,W]
    """
    # Define timesteps for DDIM
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
def expected_primitive(z0, Z, Phi_fn, log_map_fn, exp_map_fn, eps,
                       max_iter=200, lr=0.01):
    z_t = z0.clone()

    for t in range(max_iter):
        grad = torch.zeros_like(z_t)  # initialize gradient for this iteration
    
        # Loop over all samples in Z
        for i in range(Z.shape[0]):
            Phi_zt = Phi_fn(z_t)     # [1, C, H, W]
            Phi_Zi = Phi_fn(Z[i:i+1])  # [1, C, H, W], keep batch dim
    
            logs = log_map_fn(Phi_zt, Phi_Zi)  # [1, C, H, W]
    
            zt_norm2 = (z_t ** 2).sum()
            zi_norm2 = (Z[i] ** 2).sum()
    
            w = torch.exp(0.5 * (zt_norm2 - zi_norm2))  # scalar
    
            grad += 2 * w * logs  # accumulate gradient
    
        # Gradient descent step
        z_next = z_t - lr * grad
    
        # Check ε-ball constraint
        diff = z_next - z0
        if diff.view(-1).norm() >= eps:
            print(f"Stopped at iter {t} because left ε-ball.")
            break
    
        z_t = z_next
    
    return z_t

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Load model + scheduler
    # -------------------------
    model = UNetSD().to(device)
    scheduler = VPScheduler(num_timesteps=1000)

    checkpoint_path = "/data5/accounts/marsh/Diffusion/vp_diffusion_outputs/unet_epoch_2000.pt"
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded trained model.")
    else:
        print("Checkpoint not found. Using random weights.")
    model.eval()

    # -------------------------
    # Settings
    # -------------------------
    t_idx = 30
    lam = 1000.0
    beta = 0.0
    #N = 400
    eps_list = [0.01, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0]

    os.makedirs("expected_primitives", exist_ok=True)

    # -------------------------
    # Wrappers
    # -------------------------
    Phi_fn = lambda z: Phi(z, model, scheduler, num_steps=40)

    def log_map_fn(x, y):
        return log_map_shooting(
            y=x, y_target=y,
            model=model, scheduler=scheduler,
            t_idx=t_idx, lam=lam, beta=beta,
        )

    def exp_map_fn(x, v):
        return exp_map(
            x, v,
            model=model, scheduler=scheduler,
            t_idx=t_idx, lam=lam, beta=beta
        )

    # -------------------------
    # Helper: sample inside ball
    # -------------------------
    def sample_ball(z0, eps, N):
        shape = z0.shape
        d = z0.numel()
        u = torch.randn(N, *shape[1:], device=z0.device)
        u = u.view(N, -1)
        u = u / u.norm(dim=1, keepdim=True)
        r = torch.rand(N, 1, device=z0.device) ** (1.0 / d)
        samples = z0 + (eps * r * u).view(N, *shape[1:])
        return samples

    # List of sample sizes to test
    N_list = [16, 32, 64, 128, 256, 512]
    
    # Store all variances
    all_variances = {}
    
    # Loop over epsilons
    for eps in eps_list:
        all_variances[eps] = {}
        print(f"\n=== epsilon = {eps} ===")
    
        # 1. Choose center
        z0 = torch.randn(1, 3, 64, 64, device=device)
    
        # Loop over different N for variance computation
        for N_var in N_list:
            Z_var = sample_ball(z0, eps, N_var)
            print(f"Generated {Z_var.shape[0]} samples in ε-ball.")
    
            # Compute expected primitive for this N
            z_star = expected_primitive(
                z0=z0,
                Z=Z_var,
                Phi_fn=Phi_fn,
                log_map_fn=log_map_fn,
                exp_map_fn=exp_map_fn,
                eps=eps,
                max_iter=10,
                lr=0.01
            )
    
            # Compute Euclidean mean
            z_euc = Z_var.mean(dim=0, keepdim=True)
    
            # Decode for variance computation
            Φ_zstar = Phi_fn(z_star)
    
            # -------------------------
            # Compute approximate variance
            dists = []
            for i in range(N_var):
                log_i = log_map_fn(Φ_zstar, Phi_fn(Z_var[i:i+1]))
                dist_i = (log_i**2).sum().item()
                dists.append(dist_i)
            dists = torch.tensor(dists, device=device)
    
            z_star_norm = (z_star*z_star).sum()
            Z_var_norm = (Z_var*Z_var).view(N_var, -1).sum(dim=1)
            weights = torch.exp(0.5 * (z_star_norm - Z_var_norm))
    
            var_eps_N = (weights * dists).mean().item()
            all_variances[eps][N_var] = var_eps_N
            print(f"N={N_var}: Approximate variance = {var_eps_N:.6f}")
    
            # -------------------------
            # Only plot for N=64
            if N_var == 64:
                Φ_z0 = Phi_fn(z0)
                Φ_euc = Phi_fn(z_euc)
    
                # Convert to images
                def to_img(tensor):
                    return (tensor[0] * 0.5 + 0.5).clamp(0,1).cpu()
    
                imgs = [to_img(Φ_z0), to_img(Φ_zstar), to_img(Φ_euc)]
                titles = ["Center", "Expected Primitive", "Euclidean Mean"]
    
                # Plot with matplotlib
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                for ax, img, title in zip(axes, imgs, titles):
                    ax.imshow(transforms.ToPILImage()(img))
                    ax.set_title(title, fontsize=12)
                    ax.axis('off')
    
                fig.suptitle(f"ε = {eps}", fontsize=14, x=0.92, y=0.5, rotation=90, va='center')
                out_path = f"expected_primitives/epsilon_{eps}_N{N_var}_comparison.png"
                plt.tight_layout()
                plt.subplots_adjust(top=0.85, right=0.85)
                plt.savefig(out_path)
                plt.close(fig)
                print(f"Saved {out_path}")
    
    # Print summary table
    print("\n=== Variances for all epsilon balls and N ===")
    for eps in eps_list:
        print(f"\nepsilon = {eps}:")
        for N_var in N_list:
            print(f"  N={N_var}: Var(X|X in ball) = {all_variances[eps][N_var]:.6f}")


if __name__ == "__main__":
    main()
