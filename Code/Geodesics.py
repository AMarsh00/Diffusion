"""
Computes geodesics between random generations of the loaded diffusion model (CelebA-HQ and Animal Faces both work) with our metric for various values of lambda.
"""

import os
import math
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
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
            nn.Linear(time_emb_dim * 4, time_emb_dim),
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
        t = int(timesteps[i].item()) if torch.is_tensor(timesteps[i]) else int(timesteps[i])
        t_prev = int(timesteps[i + 1].item()) if torch.is_tensor(timesteps[i + 1]) else int(timesteps[i + 1])

        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[t_prev]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        t_tensor = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)
        epsilon_theta = model(x_t, t_tensor)
        x0_pred = (x_t - sqrt_one_minus_alpha_t * epsilon_theta) / sqrt_alpha_t

        sigma_t = eta * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        )
        sigma_t_val = sigma_t.item()
        noise = torch.randn_like(x_t) if sigma_t_val > 0 else torch.zeros_like(x_t)

        x_t = (
            sqrt_alpha_t_prev * x0_pred
            + torch.sqrt(1 - alpha_t_prev - sigma_t_val**2) * epsilon_theta
            + sigma_t * noise
        )

    return x_t


# ------------------------
# Fixed clean-space score proxy
# ------------------------
def score_fn(x, model, scheduler, t):
    """
    Finite-noise proxy field used by the paper:

        s_tilde(x) = -epsilon_theta(x, t_proxy) / sqrt(1 - alpha_bar_t_proxy).

    IMPORTANT: x is kept in CLEAN / generated-image coordinates x_0.
    The proxy timestep t is only supplied to the network; x is NOT first
    diffused to timestep t.
    """
    device = x.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    if isinstance(t, int):
        t_idx = int(t)
    elif torch.is_tensor(t):
        t_idx = int(t.flatten()[0].item())
    else:
        t_idx = int(t)

    t_tensor = torch.full((x.shape[0],), t_idx, device=device, dtype=torch.long)
    alpha_bar = alphas_cumprod[t_idx]
    epsilon_theta = model(x, t_tensor)
    score = -epsilon_theta / torch.sqrt(1 - alpha_bar)
    return score


def _batch_dot(a, b):
    """Per-sample Euclidean inner product, returned with broadcastable shape."""
    shape = [a.shape[0]] + [1] * (a.ndim - 1)
    return (a * b).flatten(1).sum(dim=1).view(*shape)


def score_and_jvp(x, v, model, scheduler, t_idx):
    """Return s_tilde(x) and D s_tilde(x)[v] without forming the full Jacobian."""
    x0 = x.detach()
    v0 = v.detach()

    def f(inp):
        return score_fn(inp, model, scheduler, t_idx)

    try:
        from torch.func import jvp as func_jvp
        s, jv = func_jvp(f, (x0,), (v0,))
    except Exception:
        with torch.enable_grad():
            s, jv = torch.autograd.functional.jvp(
                f,
                x0,
                v0,
                create_graph=False,
                strict=False,
            )

    return s.detach(), jv.detach()


# ------------------------
# Levi-Civita exponential map
# ------------------------
def geodesic_acceleration(x, v, model, scheduler, t_idx, metric_lambda=1.0):
    r"""
    Simplified Levi-Civita acceleration used in the paper under the
    approximately-conservative-score assumption D s_tilde ~= D s_tilde^T:

        x_ddot = -lambda/(1 + lambda ||s_tilde(x)||^2)
                  * s_tilde(x)
                  * (x_dot^T D s_tilde(x) x_dot).

    This is the rank-one metric

        g(x) = I + lambda s_tilde(x) s_tilde(x)^T.
    """
    s, jv = score_and_jvp(x, v, model, scheduler, t_idx)
    vT_J_v = _batch_dot(v, jv)
    denom = 1.0 + metric_lambda * _batch_dot(s, s)
    return -(metric_lambda / denom) * s * vT_J_v


def levi_civita_exp_map(
    x,
    v,
    model,
    scheduler,
    t_idx,
    beta=1.0,
    x_ind=None,
    n_steps=10,
    n_steps_int=10,
):
    """
    Numerical Riemannian exponential map for the CURRENT paper metric.

    The arguments x_ind and n_steps_int are retained only so old calls to this
    function still work; the path-integral metric is no longer used.

    `beta` is now the paper's metric weight lambda.  It is left in the function
    signature for compatibility with the old script.

    We integrate the second-order Levi-Civita ODE over unit time using the
    same semi-implicit Euler update stated in Algorithm 2:

        v_{k+1} = v_k + dt * a_k
        x_{k+1} = x_k + dt * v_{k+1},

    with dt = 1 / n_steps.
    """
    del x_ind, n_steps_int  # no longer part of the metric

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    x_curr = x.clone().detach()
    v_curr = v.clone().detach()
    dt = 1.0 / float(n_steps)

    for _ in range(n_steps):
        a = geodesic_acceleration(
            x_curr,
            v_curr,
            model,
            scheduler,
            t_idx,
            metric_lambda=beta,
        )
        v_curr = v_curr + dt * a
        x_curr = x_curr + dt * v_curr

    return x_curr.detach()


# ------------------------
# Log map shooting
# ------------------------
@torch.no_grad()
def log_map_shooting(
    y,
    y_target,
    model,
    scheduler,
    t_idx,
    beta=1e6,
    max_iters=24,
    lr=0.25,
    n_substeps_schedule=None,
    n_exp_steps=10,
    tol=1e-3,
):
    """
    Approximate Log_y(y_target) by damped endpoint shooting.

    This now matches the paper's Algorithm 1 much more closely:

        r <- y_target - Exp_y(v)
        v <- v + lr * r.

    No global convexity or convergence claim is made.  The best endpoint
    residual encountered is returned.

    n_substeps_schedule is retained only for compatibility with the old
    function signature and is not used by the current paper implementation.
    """
    del n_substeps_schedule

    y = y.detach()
    y_target = y_target.detach()

    v = (y_target - y).detach().clone()
    best_v = v.clone()
    best_loss = float("inf")

    for _ in range(max_iters):
        y_pred = levi_civita_exp_map(
            y,
            v,
            model,
            scheduler,
            t_idx,
            beta=beta,
            n_steps=n_exp_steps,
        )

        residual = y_target - y_pred
        loss = residual.flatten(1).pow(2).sum(dim=1).mean()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_v = v.clone()

        if torch.sqrt(loss).item() <= tol:
            break

        v = v + lr * residual

    return best_v.detach()


# ------------------------
# Image helpers
# ------------------------
def load_image(path, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


@torch.no_grad()
def Phi(y, model, scheduler, num_steps=50, eta=0.0):
    """Deterministic DDIM map Phi: z_T -> x_0."""
    timesteps = torch.linspace(
        scheduler.num_timesteps - 1,
        0,
        num_steps,
        dtype=torch.long,
        device=y.device,
    )
    x = ddim_sample(model, scheduler, y, timesteps, eta=eta)
    return x


# ------------------------
# Dataset
# ------------------------
class CelebAHQDataset(Dataset):
    def __init__(self, root_dir, image_size=64):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
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
    image_path_A = "/data5/accounts/marsh/Diffusion/celeba_hq_prepared/000100.png"  # Replace with your filepath

    x0_A = load_image(image_path_A).to(device)
    xA_ = torch.randn_like(x0_A)
    xB_ = torch.randn_like(x0_A)

    model = UNetSD().to(device)
    scheduler = VPScheduler(num_timesteps=1000)

    checkpoint_path = "/data5/accounts/marsh/Diffusion/vp_diffusion_outputs/unet_epoch_2000.pt"  # Replace with your filepath
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded trained model.")
    else:
        print("Checkpoint not found. Using random weights.")
    model.eval()

    # ------------------------------------------------------------
    # IMPORTANT NEW CONVENTION:
    # Generate the two endpoint samples x_A, x_B in CLEAN x_0 space.
    # All Riemannian geometry below is performed directly in x_0 space.
    # t_idx=400 is ONLY the network timestep used by the score proxy.
    # ------------------------------------------------------------
    xA = Phi(xA_, model, scheduler, num_steps=1000)
    xB = Phi(xB_, model, scheduler, num_steps=1000)

    # Dataset -- retained from the original file, although not needed below.
    train_dataset = CelebAHQDataset(
        root_dir="/data5/accounts/marsh/Diffusion/celeba_hq_prepared",
        image_size=64,
    )
    loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    train_images = torch.cat([b for b in loader], dim=0)
    del train_images  # retained only for compatibility with the original script

    # Proxy timestep for the metric.  This does NOT mean the geometry lives at x_400.
    t_idx = 400

    # These are values of the paper's metric parameter lambda.
    lambda_values = [0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]

    n_geo_steps = 10          # number of displayed interpolation points
    n_exp_steps = 10          # ODE steps over unit geodesic time
    shooting_iters = 24
    shooting_lr = 0.25
    shooting_tol = 1e-3

    all_geodesics = {}

    # ------------------------------------------------------------
    # Geodesics are now shot DIRECTLY from clean xA to clean xB.
    # There is no xA2/xB2 and no DDIM decoding after Exp_x(...).
    # ------------------------------------------------------------
    for lam in lambda_values:
        print(f"Computing geodesic for lambda={lam}...")

        v_forward = log_map_shooting(
            xA,
            xB,
            model,
            scheduler,
            t_idx,
            beta=lam,
            max_iters=shooting_iters,
            lr=shooting_lr,
            n_exp_steps=n_exp_steps,
            tol=shooting_tol,
        )

        # Endpoint diagnostic for the numerical shooting solve.
        endpoint_pred = levi_civita_exp_map(
            xA,
            v_forward,
            model,
            scheduler,
            t_idx,
            beta=lam,
            n_steps=n_exp_steps,
        )
        endpoint_error = (endpoint_pred - xB).flatten(1).norm(dim=1).mean().item()
        print(f"  shooting endpoint error: {endpoint_error:.6g}")

        geodesic = []
        for i in range(n_geo_steps):
            s = i / (n_geo_steps - 1)
            y_s = levi_civita_exp_map(
                xA,
                s * v_forward,
                model,
                scheduler,
                t_idx,
                beta=lam,
                n_steps=n_exp_steps,
            )

            # Pin endpoints exactly because shooting/integration are numerical.
            if i == 0:
                y_s = xA.clone()
            elif i == n_geo_steps - 1:
                y_s = xB.clone()

            # y_s is ALREADY a clean-coordinate point on the ambient geodesic.
            geodesic.append(y_s.detach().cpu())

        all_geodesics[lam] = geodesic
        print(f"Geodesic complete for lambda={lam}.")

    # ------------------------------------------------------------
    # Linear interpolation baseline in the SAME clean x_0 coordinates.
    # ------------------------------------------------------------
    linear_interp = []
    for i in range(n_geo_steps):
        s = i / (n_geo_steps - 1)
        y_s = (1 - s) * xA + s * xB
        linear_interp.append(y_s.detach().cpu())

    # Plot results
    os.makedirs("geodesic_frames", exist_ok=True)
    width, height = (64, 64)
    label_height = 15
    total_width = width * n_geo_steps
    total_height = height * len(lambda_values) + height + label_height * (len(lambda_values) + 1)
    final_image = Image.new("RGB", (total_width, total_height), color=(0, 0, 0))

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for row_idx, lam in enumerate(lambda_values):
        geodesic = all_geodesics[lam]
        y_offset = row_idx * (height + label_height)

        if font:
            draw = ImageDraw.Draw(final_image)
            draw.text((0, y_offset), f"lambda={lam}", fill=(255, 0, 0), font=font)

        y_offset += label_height

        for col_idx, frame in enumerate(geodesic):
            img = frame.squeeze(0)
            img = (img * 0.5 + 0.5).clamp(0, 1)
            pil_img = transforms.ToPILImage()(img.cpu())
            final_image.paste(pil_img, (col_idx * width, y_offset))

    # Linear interpolation row
    y_offset = len(lambda_values) * (height + label_height)
    if font:
        draw = ImageDraw.Draw(final_image)
        draw.text((0, y_offset), "LERP", fill=(255, 0, 0), font=font)
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
