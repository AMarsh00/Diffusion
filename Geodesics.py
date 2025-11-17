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
                     n_substeps_schedule=[1, 2, 4, 8, 16], train_images=None):
    y = y.detach()
    y_target = y_target.detach()
    tol = 1e-4
    #momentum_gamma = 0.9
    
    v = (y_target - y).detach().clone()
    v.requires_grad_(True)
    
    #momentum = torch.zeros_like(v)
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
            #momentum = momentum_gamma * momentum + step
            v = v + step#momentum

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
    
def main():
    device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
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
    
    xA = Phi(xA_, model, scheduler, num_steps=1000)
    xB = Phi(xB_, model, scheduler, num_steps=1000)

    # 1?Initialize the dataset
    train_dataset = CelebAHQDataset(root_dir="/data5/accounts/marsh/Diffusion/celeba_hq_prepared", image_size=64)
    
    # 2 Load all images into a single tensor
    def dataset_to_tensor(dataset, batch_size=64):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_images = []
    
        for batch in loader:
            all_images.append(batch)  # batch shape: [B, C, H, W]
    
        # Concatenate all batches
        all_images_tensor = torch.cat(all_images, dim=0)  # [N, C, H, W]
        return all_images_tensor
    
    train_dataset_tensors = dataset_to_tensor(train_dataset)  # [N, C, H, W]
    
    print(f"Loaded {train_dataset_tensors.shape[0]} images into tensor")
    
    # 3. Choose diffusion timestep for metric
    t_idx = 30#999

    # 4. Compute geodesics for multiple beta values
    beta_values = [0]#[-0.5, -0.25, 0, 0.25]
    n_geo_steps = 10
    all_geodesics = {}
    lam = 10000.0

    for beta in beta_values:
        print(f"Computing log map for beta={beta}...")
        v = log_map_shooting(
            y=xA, y_target=xB,
            model=model, scheduler=scheduler,
            t_idx=t_idx, lam=lam, beta=beta,
            max_iters=2000, lr=1e-1,
            train_images=train_dataset_tensors
        )
        print(f"Log map complete for beta={beta}.")

        geodesic = []
        for i in range(n_geo_steps):
            s = i / (n_geo_steps - 1)
            y_s = exp_map(
                xA, s * v,
                model=model, scheduler=scheduler,
                t_idx=t_idx, train_images=train_dataset_tensors,
                lam=lam, beta=beta,
            )
            x0_pred = ddim_sample(
                model, scheduler, y_s,
                torch.linspace(t_idx-1, 0, t_idx, dtype=torch.long, device=device)
            )
            geodesic.append(x0_pred.detach().cpu())
        all_geodesics[beta] = geodesic
        print(f"Geodesic computed for beta={beta}.")

    # 5. Linear interpolation in noise space
    linear_interp = []
    for i in range(n_geo_steps):
        s = i / (n_geo_steps - 1)
        y_s = (1 - s) * xA_ + s * xB_
        x_s = ddim_sample(model, scheduler, y_s, torch.linspace(999, 0, 1000, dtype=torch.long, device=device))
        linear_interp.append(x_s.detach().cpu())

    # 6. Plot all results
    from PIL import ImageDraw, ImageFont
    os.makedirs("geodesic_frames", exist_ok=True)

    # First compute width and height of individual images
    width, height = (64, 64)

    label_height = 15  # pixels to reserve for labels between rows

    total_width = width * n_geo_steps
    total_height = height * len(beta_values) + height + label_height * (len(beta_values) + 1)
    
    final_image = Image.new("RGB", (total_width, total_height), color=(0,0,0))
    
    # Optional: simple font for labels
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    for row_idx, beta in enumerate(beta_values):
        geodesic = all_geodesics[beta]
        y_offset = row_idx * (height + label_height)  # include label spacing
        # Add label above the row
        if font:
            draw = ImageDraw.Draw(final_image)
            draw.text((0, y_offset), f"beta={beta}", fill=(255,0,0), font=font)
        # Paste images below label
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
    
    output_path = "geodesic_frames/all_geodesics.png"
    final_image.save(output_path)
    print(f"Saved all geodesics and linear interpolation to {output_path}")

if __name__ == "__main__":
    main()
