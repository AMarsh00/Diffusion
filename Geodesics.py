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
    Correct VP score estimate compatible with DDPM & DDIM.
    """
    device = x.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    if isinstance(t, int):
        t_val = alphas_cumprod[t]
        t_tensor = torch.tensor([t], device=device)
    else:
        t_val = alphas_cumprod[t]
        t_tensor = t

    eps_theta = model(x, t_tensor)      # ε_θ(x_t, t)

    score = -eps_theta / torch.sqrt(1 - t_val)

    return score

# ------------------------
# Log map on image space (Euclidean approx)
# ------------------------
#@torch.no_grad()
def exp_map(x, v, model=None, scheduler=None, t_idx=None,
            lam=1000.0, beta=1.0,
            n_substeps=32):
    """
    Exponential map with conformal Riemannian metric:
        g^{-1}(x) = exp(-β * Φ(x)) * (I - λ ss^T / (1 + λ‖s‖²))
    where Φ is the accumulated line integral of <s, dx>.
    """
    if model is None:
        return x + v

    dx = v / n_substeps
    y = x.clone()
    Phi = 0.0  # accumulate ∫ s·dx

    for _ in range(n_substeps):

        # score
        s = score_fn(y, model, scheduler, t_idx)
        s_norm2 = (s * s).sum()
        proj = (s * dx).sum()

        # accumulate integral Φ += ⟨s, dx⟩
        Phi = Phi + proj.item()

        # conformal factor
        C = math.exp(-beta * Phi)

        # inverse metric action
        denom = (1 + lam * s_norm2)
        ginv_dx = dx - lam * proj * s / denom

        # apply conformal scaling
        ginv_dx = C * ginv_dx

        # update position
        y = y + ginv_dx

    return y

#@torch.no_grad()
def log_map_shooting(y, y_target, model, scheduler, t_idx,
                     lam=1000.0, max_iters=300, lr=1e-2,
                     n_substeps=32, beta=1.0):
    """
    More accurate Riemannian log map via gradient shooting.
    """
    y = y.detach()
    y_target = y_target.detach()

    v = torch.zeros_like(y, requires_grad=True)
    optimizer = torch.optim.Adam([v], lr=lr)

    for i in range(max_iters):

        y_pred = exp_map(
            y, v,
            model=model, scheduler=scheduler,
            t_idx=t_idx, lam=lam, beta=beta,
            n_substeps=n_substeps
        )

        loss = torch.norm(y_pred - y_target)**2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # optional early stop
        if loss.item() < 1e-4:
            break

    return v.detach()

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

def Phi(y, model, scheduler, num_steps=50, eta=0.0):
    """
    Map noise y -> image x using DDIM deterministic sampling.
    y: input noise tensor [B,C,H,W]
    """
    # Define timesteps for DDIM
    timesteps = torch.linspace(scheduler.num_timesteps-1, 0, num_steps, dtype=torch.long)
    x = ddim_sample(model, scheduler, y, timesteps, eta=eta)
    return x
    
def main():
    device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
    image_path_A = "/data5/accounts/marsh/Diffusion/celeba_hq_prepared/000100.png"

    x0_A = load_image(image_path_A).to(device)
    x0 = torch.randn_like(x0_A)

    model = UNetSD().to(device)
    scheduler = VPScheduler(num_timesteps=1000)
    
    checkpoint_path = "/data5/accounts/marsh/Diffusion/vp_diffusion_outputs/unet_epoch_2000.pt"
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded trained model.")
    else:
        print("Checkpoint not found. Using random weights.")
    model.eval()
    
    # 3. Choose diffusion timestep for metric
    t_idx = 0

    # 4. Compute tangent vector
    print("Computing log map...")
    v = log_map_shooting(
        y=xA, y_target=xB,
        model=model, scheduler=scheduler,
        t_idx=t_idx, lam=1000.0,
        max_iters=300, lr=1e-2,
        n_substeps=32
    )
    print("Log map complete.")

    # 5. Compute geodesic path
    print("Tracing geodesic...")
    n_geo_steps = 20
    geodesic = []
    for i in range(n_geo_steps):
        s = i / (n_geo_steps - 1)
        y_s = exp_map(
            xA, s * v,
            model=model, scheduler=scheduler,
            t_idx=t_idx, lam=1000.0,
            n_substeps=32
        )
        geodesic.append(y_s.detach().cpu())

    print("Geodesic computed.")

    # 6. Save all geodesic frames side-by-side as one wide image
    os.makedirs("geodesic_frames", exist_ok=True)

    pil_frames = []
    for frame in geodesic:
        img = frame.squeeze(0)
        img = (img * 0.5 + 0.5).clamp(0, 1)
        pil_img = transforms.ToPILImage()(img.cpu())
        pil_frames.append(pil_img)

    width, height = pil_frames[0].size
    total_width = width * len(pil_frames)

    row_image = Image.new("RGB", (total_width, height))
    x_offset = 0
    for img in pil_frames:
        row_image.paste(img, (x_offset, 0))
        x_offset += width

    output_path = "geodesic_frames/geodesic_row.png"
    row_image.save(output_path)

    print(f"Saved geodesic row to {output_path}")
    
if __name__ == "__main__":
    main()
