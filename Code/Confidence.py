"""
Computes and displays confidence scores for 16 random generations of the loaded model.
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
def _time_grid(start_t, end_t, num_steps=None):
    if start_t == end_t:
        return [int(start_t)]

    max_steps = abs(int(start_t) - int(end_t)) + 1
    if num_steps is None:
        num_steps = max_steps
    num_steps = max(2, min(int(num_steps), max_steps))

    vals = torch.linspace(start_t, end_t, num_steps).round().long().tolist()
    out = [int(vals[0])]
    for v in vals[1:]:
        v = int(v)
        if v != out[-1]:
            out.append(v)
    if out[-1] != int(end_t):
        out.append(int(end_t))
    return out


@torch.no_grad()
def ddim_sample(model, scheduler, x_T, num_steps=100, eta=0.0):
    """
    Deterministic DDIM map Phi : z_T -> x_0.

    IMPORTANT: this is the full latent-to-clean map from T to 0.  The proxy
    timestep t*=400 is NOT used to truncate Phi; it is used only to evaluate
    the fixed score proxy below.
    """
    if eta != 0.0:
        raise ValueError("This paper uses deterministic DDIM, so eta must be 0.")

    x_t = x_T.clone()
    device = x_t.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    T = scheduler.num_timesteps - 1
    timesteps = _time_grid(T, 0, num_steps)

    for t, t_prev in zip(timesteps[:-1], timesteps[1:]):
        alpha_t = alphas_cumprod[t]
        alpha_prev = alphas_cumprod[t_prev]

        t_tensor = torch.full((x_t.shape[0],), int(t), device=device, dtype=torch.long)
        epsilon_theta = model(x_t, t_tensor)

        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * epsilon_theta) / torch.sqrt(alpha_t)
        x_t = (
            torch.sqrt(alpha_prev) * x0_pred
            + torch.sqrt(1 - alpha_prev) * epsilon_theta
        )

    return x_t


def ddim_sample_differentiable(model, scheduler, x_T, num_steps=100, use_checkpoint=True):
    """Differentiable numerical DDIM map used to compute D Phi(z)^T v."""
    from torch.utils.checkpoint import checkpoint

    x_t = x_T
    device = x_t.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    T = scheduler.num_timesteps - 1
    timesteps = _time_grid(T, 0, num_steps)

    for t, t_prev in zip(timesteps[:-1], timesteps[1:]):
        alpha_t = alphas_cumprod[t]
        alpha_prev = alphas_cumprod[t_prev]

        def step(inp, t=t, alpha_t=alpha_t, alpha_prev=alpha_prev):
            t_tensor = torch.full(
                (inp.shape[0],), int(t), device=inp.device, dtype=torch.long
            )
            epsilon_theta = model(inp, t_tensor)
            x0_pred = (
                inp - torch.sqrt(1 - alpha_t) * epsilon_theta
            ) / torch.sqrt(alpha_t)
            return (
                torch.sqrt(alpha_prev) * x0_pred
                + torch.sqrt(1 - alpha_prev) * epsilon_theta
            )

        if use_checkpoint and x_t.requires_grad:
            x_t = checkpoint(step, x_t, use_reentrant=False)
        else:
            x_t = step(x_t)

    return x_t


# ------------------------
# Fixed clean-space score proxy / Riemannian metric
# ------------------------
def score_proxy(x_clean, model, scheduler, t_idx):
    r"""
    Fixed finite-noise proxy field used by the paper:

        s_tilde(x) = -epsilon_theta(x, t*) / sqrt(1-alpha_bar_t*).

    x remains a CLEAN x_0-space coordinate.  We do NOT diffuse x to timestep
    t* before evaluating the network.
    """
    device = x_clean.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    alpha_bar = alphas_cumprod[t_idx]
    t_tensor = torch.full(
        (x_clean.shape[0],), int(t_idx), device=device, dtype=torch.long
    )
    epsilon_theta = model(x_clean, t_tensor)
    return -epsilon_theta / torch.sqrt(1 - alpha_bar)


def _batch_dot(a, b):
    shape = [a.shape[0]] + [1] * (a.ndim - 1)
    return (a * b).flatten(1).sum(dim=1).view(*shape)


def metric_apply_from_score(s, v, metric_lambda):
    """Apply g(x)=I+lambda s s^T to v without forming the matrix."""
    return v + metric_lambda * s * _batch_dot(s, v)


def metric_squared_norm_from_score(s, v, metric_lambda):
    """
    Return ||v||_g^2 = ||v||_2^2 + lambda (s^T v)^2 for each batch item.

    The scalar reductions are accumulated in float64.  This does not change the
    metric; it only prevents float32 overflow when lambda and the ambient
    dimension are large.
    """
    s64 = s.double()
    v64 = v.double()
    euclid = v64.flatten(1).pow(2).sum(dim=1)
    score_dot = (s64 * v64).flatten(1).sum(dim=1)
    return euclid + float(metric_lambda) * score_dot.pow(2)


def score_proxy_and_jvp(x, v, model, scheduler, t_idx):
    """Return s_tilde(x) and D s_tilde(x)[v] without forming the full Jacobian."""
    x0 = x.detach()
    v0 = v.detach()

    def fn(inp):
        return score_proxy(inp, model, scheduler, t_idx)

    try:
        from torch.func import jvp as func_jvp
        s, jv = func_jvp(fn, (x0,), (v0,))
    except Exception:
        with torch.enable_grad():
            s, jv = torch.autograd.functional.jvp(
                fn, x0, v0, create_graph=False, strict=False
            )

    return s.detach(), jv.detach()


# ------------------------
# Levi-Civita Exponential Map
# ------------------------
def geodesic_acceleration(x, v, model, scheduler, t_idx, metric_lambda):
    r"""
    Simplified Levi-Civita acceleration used in the paper under
    D s_tilde ~= D s_tilde^T:

        x_ddot = -lambda/(1+lambda||s||^2)
                    s (x_dot^T D s x_dot).
    """
    s, jv = score_proxy_and_jvp(x, v, model, scheduler, t_idx)
    v_j_v = _batch_dot(v, jv)
    denom = 1.0 + metric_lambda * _batch_dot(s, s)
    return -(metric_lambda / denom) * s * v_j_v


def levi_civita_exp_map(
    x,
    v,
    model,
    scheduler,
    t_idx,
    metric_lambda=1.0,
    n_steps=6,
):
    """Numerically integrate the paper's geodesic ODE over unit time."""
    x_curr = x.clone().detach()
    v_curr = v.clone().detach()
    dt = 1.0 / float(n_steps)

    for _ in range(n_steps):
        a = geodesic_acceleration(
            x_curr, v_curr, model, scheduler, t_idx, metric_lambda
        )
        # Semi-implicit Euler, matching Algorithm 2.
        v_curr = v_curr + dt * a
        x_curr = x_curr + dt * v_curr

    return x_curr.detach()


# ------------------------
# Levi-Civita Log Map Shooting
# ------------------------
def _endpoint_residual_stats_batch(y_pred, y_target):
    """Per-sample endpoint residual, L2 error, and per-coordinate RMSE."""
    residual = y_target - y_pred
    flat = residual.flatten(1)
    l2 = flat.norm(dim=1)
    rmse = flat.pow(2).mean(dim=1).sqrt()
    return residual, l2, rmse


@torch.no_grad()
def levi_civita_log_map(
    Y,
    Y_target,
    model,
    scheduler,
    t_idx,
    metric_lambda=1.0,
    n_substeps_schedule=(1, 2, 4, 8, 16),
    max_iters=20,
    lr=0.5,
    tol=1e-3,
    initial_v=None,
    backtracking_factor=0.5,
    max_backtracking_steps=4,
    lr_growth=1.10,
    min_lr=1e-5,
    max_lr=1.0,
    return_diagnostics=False,
):
    r"""
    Vectorized coarse-to-fine endpoint shooting for Log_Y(Y_target).

    The geometry is unchanged.  At every fixed ODE resolution S the shooting
    correction remains the paper's residual direction

        r <- Y_target - Exp_Y(v),
        v <- v + eta r.

    Numerical safeguards:

      * coarse-to-fine continuation in the exponential-map ODE resolution;
      * per-sample monotone backtracking on endpoint RMSE;
      * the best tangent at each resolution initializes the next resolution;
      * stopping is based on per-coordinate RMSE, not raw high-dimensional L2.

    This is the same shooting strategy used by the corrected standalone
    geodesic code, vectorized for the many local log maps required by the
    confidence estimator.
    """
    if not (0.0 < backtracking_factor < 1.0):
        raise ValueError("backtracking_factor must lie in (0,1)")
    if max_backtracking_steps < 1:
        raise ValueError("max_backtracking_steps must be >= 1")
    if lr <= 0:
        raise ValueError("lr must be positive")

    schedule = tuple(int(s) for s in n_substeps_schedule)
    if len(schedule) == 0 or any(s < 1 for s in schedule):
        raise ValueError("n_substeps_schedule must contain positive integers")

    # Remove accidental consecutive duplicates.
    cleaned = []
    for s in schedule:
        if not cleaned or s != cleaned[-1]:
            cleaned.append(s)
    schedule = tuple(cleaned)

    if max_iters < len(schedule):
        raise ValueError(
            "max_iters must be at least the number of shooting substep stages"
        )

    Y = Y.detach().clone()
    Y_target = Y_target.detach().clone()
    N = Y.shape[0]
    view_shape = [N] + [1] * (Y.ndim - 1)

    if initial_v is None:
        v = (Y_target - Y).clone()
    else:
        v = initial_v.detach().clone()
        if v.shape != Y.shape:
            raise ValueError("initial_v must have the same shape as Y")

    # Allocate more of the total iteration budget to finer ODE resolutions.
    n_stages = len(schedule)
    stage_weights = torch.arange(1, n_stages + 1, dtype=torch.float64)
    raw_alloc = float(max_iters) * stage_weights / stage_weights.sum()
    iters_per_stage = [max(1, int(round(x.item()))) for x in raw_alloc]

    diff = int(max_iters - sum(iters_per_stage))
    idx = n_stages - 1
    while diff != 0:
        if diff > 0:
            iters_per_stage[idx] += 1
            diff -= 1
        elif iters_per_stage[idx] > 1:
            iters_per_stage[idx] -= 1
            diff += 1
        idx = (idx - 1) % n_stages

    final_best_v = v.clone()
    final_best_rmse = torch.full(
        (N,), float("inf"), device=Y.device, dtype=torch.float64
    )
    final_best_l2 = torch.full_like(final_best_rmse, float("inf"))

    for stage_idx, (n_steps, stage_iters) in enumerate(zip(schedule, iters_per_stage)):
        # Re-evaluate the incoming tangent using this stage's endpoint map.
        Y_pred = levi_civita_exp_map(
            Y,
            v,
            model,
            scheduler,
            t_idx,
            metric_lambda=metric_lambda,
            n_steps=n_steps,
        )
        residual, l2, rmse = _endpoint_residual_stats_batch(Y_pred, Y_target)

        finite = (
            torch.isfinite(Y_pred).flatten(1).all(dim=1)
            & torch.isfinite(rmse)
        )
        rmse64 = rmse.double()
        l264 = l2.double()
        rmse64 = torch.where(
            finite, rmse64, torch.full_like(rmse64, float("inf"))
        )
        l264 = torch.where(
            finite, l264, torch.full_like(l264, float("inf"))
        )

        stage_best_v = v.clone()
        stage_best_rmse = rmse64.clone()
        stage_best_l2 = l264.clone()

        # One adaptive eta per log-map pair.  This is important because a batch
        # can contain both easy and difficult local endpoint pairs.
        eta = torch.full(
            (N,), float(lr), device=Y.device, dtype=Y.dtype
        )

        for _ in range(stage_iters):
            # Save any improvements before proposing another correction.
            better = rmse64 < stage_best_rmse
            if better.any():
                stage_best_v = torch.where(
                    better.view(*view_shape), v, stage_best_v
                )
                stage_best_rmse = torch.minimum(stage_best_rmse, rmse64)
                stage_best_l2 = torch.where(better, l264, stage_best_l2)

            active = torch.isfinite(rmse64) & (rmse64 > float(tol))
            if not active.any():
                break

            # Each sample gets an independent backtracking step length.
            trial_eta = eta.clone()
            accepted_this_iter = torch.zeros(N, dtype=torch.bool, device=Y.device)

            for _bt in range(max_backtracking_steps):
                bt_active = active & (~accepted_this_iter)
                if not bt_active.any():
                    break

                v_trial = v + trial_eta.view(*view_shape) * residual
                finite_v = torch.isfinite(v_trial).flatten(1).all(dim=1)

                # Only evaluate finite candidates.  For simplicity and GPU
                # efficiency we still execute the vectorized exp map on the
                # whole batch, but replace bad trial tangents by the current v.
                safe_trial = torch.where(
                    finite_v.view(*view_shape), v_trial, v
                )
                Y_trial = levi_civita_exp_map(
                    Y,
                    safe_trial,
                    model,
                    scheduler,
                    t_idx,
                    metric_lambda=metric_lambda,
                    n_steps=n_steps,
                )
                residual_trial, l2_trial, rmse_trial = _endpoint_residual_stats_batch(
                    Y_trial, Y_target
                )

                finite_trial = (
                    finite_v
                    & torch.isfinite(Y_trial).flatten(1).all(dim=1)
                    & torch.isfinite(rmse_trial)
                )
                trial_rmse64 = rmse_trial.double()

                improve = (
                    bt_active
                    & finite_trial
                    & (trial_rmse64 < rmse64 - 1e-10)
                )

                if improve.any():
                    v = torch.where(improve.view(*view_shape), v_trial, v)
                    residual = torch.where(
                        improve.view(*view_shape), residual_trial, residual
                    )
                    rmse64 = torch.where(improve, trial_rmse64, rmse64)
                    l264 = torch.where(improve, l2_trial.double(), l264)

                    eta_new = torch.clamp(
                        trial_eta * float(lr_growth),
                        min=float(min_lr),
                        max=float(max_lr),
                    )
                    eta = torch.where(improve, eta_new, eta)
                    accepted_this_iter |= improve

                rejected = bt_active & (~improve)
                if rejected.any():
                    trial_eta = torch.where(
                        rejected,
                        trial_eta * float(backtracking_factor),
                        trial_eta,
                    )

            # Pairs that found no improving residual correction at this
            # resolution simply keep their best tangent and continue to the
            # next, finer endpoint map.
            if not accepted_this_iter.any():
                break

        # Include the final iterate in the stage-best comparison.
        better = rmse64 < stage_best_rmse
        if better.any():
            stage_best_v = torch.where(
                better.view(*view_shape), v, stage_best_v
            )
            stage_best_rmse = torch.minimum(stage_best_rmse, rmse64)
            stage_best_l2 = torch.where(better, l264, stage_best_l2)

        # Coarse-to-fine continuation uses the best tangent at this resolution.
        v = stage_best_v.clone()

        if stage_idx == n_stages - 1:
            final_best_v = stage_best_v.clone()
            final_best_rmse = stage_best_rmse.clone()
            final_best_l2 = stage_best_l2.clone()

    if return_diagnostics:
        finite = torch.isfinite(final_best_rmse)
        if finite.any():
            vals = final_best_rmse[finite]
            l2vals = final_best_l2[finite]
            diagnostics = {
                "mean_rmse": float(vals.mean().item()),
                "median_rmse": float(vals.median().item()),
                "max_rmse": float(vals.max().item()),
                "mean_l2": float(l2vals.mean().item()),
                "finite_fraction": float(finite.double().mean().item()),
                "final_ode_steps": int(schedule[-1]),
            }
        else:
            diagnostics = {
                "mean_rmse": float("inf"),
                "median_rmse": float("inf"),
                "max_rmse": float("inf"),
                "mean_l2": float("inf"),
                "finite_fraction": 0.0,
                "final_ode_steps": int(schedule[-1]),
            }
        return final_best_v.detach(), diagnostics

    return final_best_v.detach()


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


# ------------------------
# Confidence Metric
# ------------------------
def confidence_metric(
    z,
    Phi_fn,
    Phi_grad_fn,
    log_map_fn,
    model,
    scheduler,
    t_idx,
    metric_lambda,
    eps=0.1,
    N=64,
    max_iter=2,
    lr=0.05,
    delta=1e-5,
    backtracking_factor=0.5,
    max_backtracking_steps=4,
):
    r"""
    Compute the paper's generator-constrained local geometric confidence score.

    The mathematics is unchanged from the revised paper.  The important
    numerical change from the previous version is that the latent Frechet
    direction is NORMALIZED and accepted with a monotone line search on the
    same fixed Monte Carlo objective.  The unnormalized VJP can be enormous in
    3x64x64 dimensions even when its direction is correct; multiplying it
    directly by lr was the source of the inf values.
    """
    if z.shape[0] != 1:
        raise ValueError("confidence_metric currently expects batch size 1")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not (0.0 < backtracking_factor < 1.0):
        raise ValueError("backtracking_factor must lie in (0,1)")

    device = z.device

    # ------------------------------------------------------------
    # 1. Fixed proposal Z_i ~ N(z0, eps^2 I) and fixed normalized weights.
    # ------------------------------------------------------------
    with torch.no_grad():
        Z = z + float(eps) * torch.randn(
            N, *z.shape[1:], device=device, dtype=z.dtype
        )

        # Accumulate latent norms in float64.  Softmax is shift-stable, but
        # float64 also avoids loss of resolution in this very high dimension.
        log_w = -0.5 * Z.double().flatten(1).pow(2).sum(dim=1)
        log_w = log_w - log_w.max()
        weights = torch.softmax(log_w, dim=0)  # float64, sums to one

        Phi_Z = Phi_fn(Z)
        if not torch.isfinite(Phi_Z).all():
            raise FloatingPointError(
                "Phi(Z) became non-finite. Reduce phi_steps or check the DDIM/model checkpoint."
            )

    def fixed_energy_from_x(x_candidate):
        """Same fixed empirical Frechet objective used by the line search."""
        x_rep = x_candidate.expand(N, -1, -1, -1)
        logs_here = log_map_fn(x_rep, Phi_Z)
        if not torch.isfinite(logs_here).all():
            return torch.tensor(float("inf"), device=device, dtype=torch.float64)
        s_here = score_proxy(x_candidate, model, scheduler, t_idx)
        s_rep = s_here.expand(N, -1, -1, -1)
        d2_here = metric_squared_norm_from_score(
            s_rep, logs_here, metric_lambda
        )
        return (weights * d2_here).sum()

    # ------------------------------------------------------------
    # 2. Generator-constrained local Frechet mean in latent space.
    # ------------------------------------------------------------
    z_t = z.detach().clone()

    for _ in range(max_iter):
        with torch.enable_grad():
            z_req = z_t.detach().clone().requires_grad_(True)
            Phi_zt_graph = Phi_grad_fn(z_req)

        if not torch.isfinite(Phi_zt_graph).all():
            raise FloatingPointError(
                "Differentiable Phi(z) became non-finite before the Frechet update."
            )

        Phi_zt = Phi_zt_graph.detach()
        Phi_zt_expanded = Phi_zt.expand(N, -1, -1, -1)
        logs = log_map_fn(Phi_zt_expanded, Phi_Z)
        if not torch.isfinite(logs).all():
            raise FloatingPointError(
                "A local log map became non-finite. Increase the final shooting substep resolution or reduce shooting lr."
            )

        # Baseline objective from these same fixed logs/candidates/weights.
        with torch.no_grad():
            s = score_proxy(Phi_zt, model, scheduler, t_idx)
            s_rep = s.expand(N, -1, -1, -1)
            baseline_d2 = metric_squared_norm_from_score(
                s_rep, logs, metric_lambda
            )
            baseline_energy = (weights * baseline_d2).sum()

        # Delta = sum_i w_i Log_x(Y_i).  Cast only the normalized weights back
        # to the image dtype for the vector average.
        wview = weights.to(logs.dtype).view(N, 1, 1, 1)
        delta_x = (wview * logs).sum(dim=0, keepdim=True)

        # g(x) Delta.  This covector can have a huge norm when lambda is large.
        # Rescaling it by a positive scalar does NOT change the descent
        # direction D Phi^T g Delta, and prevents overflow in the VJP.
        with torch.no_grad():
            pullback_covector = metric_apply_from_score(
                s, delta_x, metric_lambda
            ).detach()
            cov_norm = pullback_covector.flatten(1).norm(dim=1).clamp_min(1e-12)
            if not torch.isfinite(cov_norm).all():
                raise FloatingPointError("g(x)Delta became non-finite")
            pullback_covector = pullback_covector / cov_norm.view(1, 1, 1, 1)

        with torch.enable_grad():
            scalar = (Phi_zt_graph * pullback_covector).sum()
            latent_dir = torch.autograd.grad(
                scalar, z_req, retain_graph=False, create_graph=False
            )[0].detach()

        if not torch.isfinite(latent_dir).all():
            raise FloatingPointError(
                "D Phi(z)^T g(x)Delta became non-finite. Try phi_steps=64 or 40."
            )

        # Only the DIRECTION matters for descent.  The previous file multiplied
        # this potentially enormous vector directly by lr; now lr is the total
        # latent L2 step length, exactly like the normalized refinement code.
        dir_norm = latent_dir.flatten(1).norm(dim=1).clamp_min(1e-12)
        if dir_norm.item() <= 1e-12:
            break
        unit_dir = latent_dir / dir_norm.view(1, 1, 1, 1)

        accepted = False
        trial_step = float(lr)

        for _ in range(max_backtracking_steps):
            trial_z = z_t + trial_step * unit_dir
            with torch.no_grad():
                trial_x = Phi_fn(trial_z)
                if torch.isfinite(trial_x).all():
                    trial_energy = fixed_energy_from_x(trial_x)
                else:
                    trial_energy = torch.tensor(
                        float("inf"), device=device, dtype=torch.float64
                    )

            if torch.isfinite(trial_energy) and trial_energy <= baseline_energy:
                z_t = trial_z.detach()
                accepted = True
                break

            trial_step *= float(backtracking_factor)

        if not accepted:
            # Current z_t is already finite and the line search found no
            # improving finite step.  Keeping it is the correct stable action.
            break

        if trial_step < 1e-4:
            break

    z_star = z_t

    # ------------------------------------------------------------
    # 3. Mean, variance, and center-to-mean distance.
    # ------------------------------------------------------------
    with torch.no_grad():
        x_star = Phi_fn(z_star)
        x_center = Phi_fn(z)
        if not torch.isfinite(x_star).all() or not torch.isfinite(x_center).all():
            raise FloatingPointError("Final Phi evaluation became non-finite")

        x_star_expanded = x_star.expand(N, -1, -1, -1)
        logs_var = log_map_fn(x_star_expanded, Phi_Z)
        if not torch.isfinite(logs_var).all():
            raise FloatingPointError("Variance log maps became non-finite")

        s_star = score_proxy(x_star, model, scheduler, t_idx)
        s_star_expanded = s_star.expand(N, -1, -1, -1)
        d2_var = metric_squared_norm_from_score(
            s_star_expanded, logs_var, metric_lambda
        )
        var_R = (weights * d2_var).sum()  # float64

        log_center = log_map_fn(x_center, x_star)
        if not torch.isfinite(log_center).all():
            raise FloatingPointError("Center-to-mean log map became non-finite")

        s_center = score_proxy(x_center, model, scheduler, t_idx)
        dist_center_sq = metric_squared_norm_from_score(
            s_center, log_center, metric_lambda
        )[0]  # float64

        delta64 = torch.tensor(float(delta), device=device, dtype=torch.float64)
        C = torch.log(var_R + delta64) + dist_center_sq / (var_R + delta64)
        ess = 1.0 / weights.pow(2).sum().clamp_min(1e-30)

        if not torch.isfinite(C):
            raise FloatingPointError(
                f"Confidence remained non-finite: V={var_R.item()}, d2={dist_center_sq.item()}"
            )

    return (
        float(C.item()),
        float(var_R.item()),
        float(dist_center_sq.item()),
        float(ess.item()),
        z_star.detach(),
        x_star.detach(),
    )


# ------------------------
# Main
# ------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Geometry / confidence parameters.
    t_idx = 400                  # ONLY the score-proxy timestep.
    metric_lambda = 10000.0      # Same role as lambda in g = I + lambda ss^T.
    num_samples = 16
    eps = 0.1

    # Numerical parameters.
    phi_steps = 100              # Increase toward 1000 for final high-fidelity runs.
    display_steps = 1000         # High-fidelity rendering only.
    # Local log maps use the same coarse-to-fine shooting strategy that fixed
    # the standalone geodesic solve.  These pairs are much more local than the
    # global endpoint geodesic, so we stop at 16 ODE steps rather than 32.
    local_shooting_substeps = (1, 2, 4, 8, 16)
    local_shooting_iters = 20      # TOTAL iterations across all substep stages.
    local_shooting_lr = 0.5
    local_shooting_tol = 1e-3      # per-coordinate endpoint RMSE
    local_shooting_backtracking_factor = 0.5
    local_shooting_max_backtracking_steps = 4
    N = 64
    mean_iters = 2
    mean_lr = 0.05           # TOTAL latent L2 step after direction normalization.
    variance_floor = 1e-5
    mean_backtracking_factor = 0.5
    mean_max_backtracking_steps = 4

    # Load model & scheduler.
    model = UNetSD().to(device)
    scheduler = VPScheduler(num_timesteps=1000)
    checkpoint_path = "/data5/accounts/marsh/Diffusion/vp_diffusion_outputs/unet_epoch_2000.pt"

    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded trained model.")
    else:
        raise FileNotFoundError(checkpoint_path)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Same numerical Phi is used throughout the confidence calculation.
    @torch.no_grad()
    def Phi_fn(z):
        return ddim_sample(
            model, scheduler, z, num_steps=phi_steps, eta=0.0
        )

    def Phi_grad_fn(z):
        return ddim_sample_differentiable(
            model, scheduler, z, num_steps=phi_steps, use_checkpoint=True
        )

    @torch.no_grad()
    def Phi_display_fn(z):
        return ddim_sample(
            model, scheduler, z, num_steps=display_steps, eta=0.0
        )

    shooting_diagnostics = []

    def log_map_fn(x, y):
        logs, stats = levi_civita_log_map(
            Y=x,
            Y_target=y,
            model=model,
            scheduler=scheduler,
            t_idx=t_idx,
            metric_lambda=metric_lambda,
            n_substeps_schedule=local_shooting_substeps,
            max_iters=local_shooting_iters,
            lr=local_shooting_lr,
            tol=local_shooting_tol,
            backtracking_factor=local_shooting_backtracking_factor,
            max_backtracking_steps=local_shooting_max_backtracking_steps,
            return_diagnostics=True,
        )
        shooting_diagnostics.append(stats)
        return logs

    # Generate batch of random z_T latent variables.
    z0_batch = torch.randn(num_samples, 3, 64, 64, device=device)

    C_all = []
    V_all = []
    D_all = []
    ESS_all = []

    for i in range(num_samples):
        z0 = z0_batch[i:i + 1]
        print(f"Computing confidence score {i + 1}/{num_samples}...")

        diag_start = len(shooting_diagnostics)

        C, var_R, dist_sq, ess, _, _ = confidence_metric(
            z=z0,
            Phi_fn=Phi_fn,
            Phi_grad_fn=Phi_grad_fn,
            log_map_fn=log_map_fn,
            model=model,
            scheduler=scheduler,
            t_idx=t_idx,
            metric_lambda=metric_lambda,
            eps=eps,
            N=N,
            max_iter=mean_iters,
            lr=mean_lr,
            delta=variance_floor,
            backtracking_factor=mean_backtracking_factor,
            max_backtracking_steps=mean_max_backtracking_steps,
        )

        C_all.append(C)
        V_all.append(var_R)
        D_all.append(dist_sq)
        ESS_all.append(ess)

        print(
            f"  C={C:.6g}, variance={var_R:.6g}, "
            f"distance^2={dist_sq:.6g}, ESS={ess:.2f}/{N}"
        )

        # Summarize all log-map shooting solves used for this confidence score.
        new_diags = shooting_diagnostics[diag_start:]
        if new_diags:
            mean_rmse = float(np.mean([d["mean_rmse"] for d in new_diags]))
            max_rmse = float(np.max([d["max_rmse"] for d in new_diags]))
            finite_fraction = float(np.mean([d["finite_fraction"] for d in new_diags]))
            print(
                f"  local log shooting: mean RMSE={mean_rmse:.6g}, "
                f"max RMSE={max_rmse:.6g}, finite={finite_fraction:.3f}"
            )

    C_all = np.asarray(C_all)
    V_all = np.asarray(V_all)
    D_all = np.asarray(D_all)
    ESS_all = np.asarray(ESS_all)

    print("Confidence scores:", C_all)
    print("Local variances:", V_all)
    print("Squared distances to local Frechet means:", D_all)
    print("Importance ESS:", ESS_all)

    # ------------------------
    # Generate final DDIM samples for visualization
    # ------------------------
    samples = Phi_display_fn(z0_batch)
    samples = (samples * 0.5 + 0.5).clamp(0, 1)

    # Plot 4x4 grid (for 16 samples).
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(num_samples):
        ax = axes[i // 4, i % 4]
        img = transforms.ToPILImage()(samples[i].cpu())
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Score: {C_all[i]:.4f}", fontsize=10)

    plt.tight_layout()
    os.makedirs("confidences", exist_ok=True)
    out_path = "confidences/epsilon_batch_samples_corrected.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved batch plot to {out_path}")


if __name__ == "__main__":
    main()
