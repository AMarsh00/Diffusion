"""
High-Confidence Curves -- paper-consistent Route-2 implementation.

This file intentionally preserves the finite-sample HCC behavior that produced
the earlier image results while targeting the SAME localized measure used in
the manuscript.

Geometry
--------
The exact score-induced metric is

    g(x) = I + lambda s_tilde(x) s_tilde(x)^T.

For image-scale computation, this implementation uses the symmetry-reduced
paper Eq. (8) flow

    x_ddot = -lambda/(1 + lambda ||s_tilde||^2)
              s_tilde (x_dot^T D s_tilde x_dot)

as an explicitly defined computational surrogate.  It is not claimed to equal
the full Eq. (7) Levi-Civita dynamics for a nonsymmetric learned score Jacobian.

Localized measure
-----------------
The manuscript's theoretical localized law is

    nu_{z0,sigma}(dz) proportional to
        exp(-||z-z0||^2/(2 sigma^2)) mu(dz),

with mu=N(0,I).  Equivalently,

    nu_{z0,sigma}
      = N(z0/(1+sigma^2), sigma^2/(1+sigma^2) I).

Although this law can be sampled directly, the production HCC code below uses
the ORIGINAL proposal

    q_{z0,sigma} = N(z0, sigma^2 I)

with self-normalized importance weights

    w_j proportional to mu(Z_j) = exp(-||Z_j||^2/2).

This does NOT change the theoretical target measure: since the density of q is
proportional to the localization kernel K_sigma(.,z0), the importance ratio
nu/q is proportional to mu.  We retain this estimator because it preserves the
finite-sample numerical behavior of the earlier HCC experiments.  A direct
sampler for the same target measure is included below for validation, but is
not the default production estimator.

Curve nodes remain generator-constrained through the deterministic DDIM map
Phi.  Local surrogate-log directions are pulled back through the exact
numerical DDIM VJP D Phi(z)^T, projected perpendicular to the discrete latent
curve, and accepted only after per-node backtracking on the same frozen
importance-sampled local objective.
"""

import os
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# UNet -- unchanged for checkpoint compatibility
# -----------------------------------------------------------------------------
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
        return torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_skip_conv=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.skip_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels or use_skip_conv
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        h = self.conv1(self.relu(self.norm1(x)))
        h = h + self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.relu(self.norm2(h)))
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
            ResidualBlock(base_channels, base_channels * 2, time_emb_dim, True),
            ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim, True),
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
            ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim, True),
            ResidualBlock(base_channels * 4, base_channels, time_emb_dim, True),
            ResidualBlock(base_channels * 2, base_channels, time_emb_dim, True),
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
            x = torch.cat([x, residuals.pop()], dim=1)
            x = block(x, t_emb)
        return self.out_conv(self.out_relu(self.out_norm(x)))


# -----------------------------------------------------------------------------
# VP schedule and deterministic DDIM maps
# -----------------------------------------------------------------------------
class VPScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)


def _time_grid(start_t: int, end_t: int, num_steps: Optional[int] = None):
    if start_t == end_t:
        return [int(start_t)]
    max_steps = abs(start_t - end_t) + 1
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
def ddim_reverse_segment(model, scheduler, x_start, start_t, end_t, num_steps=None):
    """Numerical deterministic DDIM map from a noisier state to a cleaner state."""
    if start_t < end_t:
        raise ValueError("ddim_reverse_segment requires start_t >= end_t")
    if start_t == end_t:
        return x_start.clone()

    x = x_start.clone()
    alpha_cum = scheduler.alphas_cumprod.to(x.device)
    grid = _time_grid(start_t, end_t, num_steps)
    for t, t_prev in zip(grid[:-1], grid[1:]):
        alpha_t = alpha_cum[t]
        alpha_prev = alpha_cum[t_prev]
        tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        eps = model(x, tt)
        x0_pred = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
        x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * eps
    return x


@torch.no_grad()
def ddim_forward_inversion_segment(model, scheduler, x_start, start_t, end_t, num_steps=None):
    """Numerical deterministic DDIM inversion from a cleaner state to a noisier state."""
    if start_t > end_t:
        raise ValueError("ddim_forward_inversion_segment requires start_t <= end_t")
    if start_t == end_t:
        return x_start.clone()

    x = x_start.clone()
    alpha_cum = scheduler.alphas_cumprod.to(x.device)
    grid = _time_grid(start_t, end_t, num_steps)
    for t, t_next in zip(grid[:-1], grid[1:]):
        alpha_t = alpha_cum[t]
        alpha_next = alpha_cum[t_next]
        tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        eps = model(x, tt)
        x0_pred = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
        x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * eps
    return x


def ddim_reverse_segment_differentiable(
    model,
    scheduler,
    x_start,
    start_t,
    end_t,
    num_steps=None,
    use_checkpoint=True,
):
    """Differentiable numerical DDIM map, used for D Phi(z)^T covector VJPs."""
    if start_t < end_t:
        raise ValueError("start_t must be >= end_t")
    if start_t == end_t:
        return x_start

    from torch.utils.checkpoint import checkpoint

    x = x_start
    alpha_cum = scheduler.alphas_cumprod.to(x.device)
    grid = _time_grid(start_t, end_t, num_steps)

    for t, t_prev in zip(grid[:-1], grid[1:]):
        alpha_t = alpha_cum[t]
        alpha_prev = alpha_cum[t_prev]

        def step(inp, t=t, alpha_t=alpha_t, alpha_prev=alpha_prev):
            tt = torch.full((inp.shape[0],), int(t), device=inp.device, dtype=torch.long)
            eps = model(inp, tt)
            x0_pred = (inp - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            return torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * eps

        if use_checkpoint and x.requires_grad:
            x = checkpoint(step, x, use_reentrant=False)
        else:
            x = step(x)
    return x


# -----------------------------------------------------------------------------
# FIXED CLEAN-SPACE SCORE PROXY
# -----------------------------------------------------------------------------
def score_proxy(x_clean, model, scheduler, proxy_t: int):
    r"""
    Numerically stable proxy field used by the paper.

    IMPORTANT: x_clean remains an x_0-space coordinate.  proxy_t is only the
    network timestep used to evaluate the surrogate field; x_clean is NOT first
    noised to x_proxy_t.
    """
    device = x_clean.device
    alpha_bar = scheduler.alphas_cumprod.to(device)[proxy_t]
    tt = torch.full((x_clean.shape[0],), int(proxy_t), device=device, dtype=torch.long)
    epsilon_theta = model(x_clean, tt)
    return -epsilon_theta / torch.sqrt(1.0 - alpha_bar)


def _batch_dot(a, b):
    shape = [a.shape[0]] + [1] * (a.ndim - 1)
    return (a * b).flatten(1).sum(dim=1).view(*shape)


def metric_inner_from_score(s, u, v, metric_lambda):
    return _batch_dot(u, v) + metric_lambda * _batch_dot(s, u) * _batch_dot(s, v)


def metric_apply_from_score(s, v, metric_lambda):
    """Apply g(x)=I+lambda ss^T to a tangent vector."""
    return v + metric_lambda * s * _batch_dot(s, v)


def score_proxy_and_jvp(x, v, model, scheduler, proxy_t):
    """Return s_tilde(x) and D s_tilde(x)[v] without forming the full Jacobian."""
    x0 = x.detach()
    v0 = v.detach()

    def f(inp):
        return score_proxy(inp, model, scheduler, proxy_t)

    try:
        from torch.func import jvp as func_jvp
        s, jv = func_jvp(f, (x0,), (v0,))
    except Exception:
        with torch.enable_grad():
            s, jv = torch.autograd.functional.jvp(
                f, x0, v0, create_graph=False, strict=False
            )
    return s.detach(), jv.detach()


def geodesic_acceleration_proxy(x, v, model, scheduler, proxy_t, metric_lambda):
    r"""
    Symmetry-reduced computational surrogate from paper Eq. (8):

      a = -lambda/(1+lambda||s_tilde||^2)
            s_tilde (v^T D s_tilde v).

    This equals the exact Levi-Civita acceleration only when D s_tilde is
    symmetric.  For the learned image model it is used deliberately as the
    stable Route-2 surrogate flow, not as a claim that Eq. (7) is being
    integrated.
    """
    s, jv = score_proxy_and_jvp(x, v, model, scheduler, proxy_t)
    v_j_v = _batch_dot(v, jv)
    denom = 1.0 + metric_lambda * _batch_dot(s, s)
    return -(metric_lambda / denom) * s * v_j_v


def geodesic_exp_map_proxy(
    x,
    v,
    model,
    scheduler,
    proxy_t,
    metric_lambda=1.0,
    n_steps=6,
):
    """Integrate the symmetry-reduced Eq. (8) surrogate flow over unit time."""
    x_curr = x.detach().clone()
    v_curr = v.detach().clone()
    dt = 1.0 / float(n_steps)
    for _ in range(n_steps):
        a = geodesic_acceleration_proxy(
            x_curr, v_curr, model, scheduler, proxy_t, metric_lambda
        )
        v_curr = v_curr + dt * a
        x_curr = x_curr + dt * v_curr
    return x_curr.detach()


# Preferred Route-2 names.  Old names are retained for compatibility with the
# existing experiment scripts and checkpoints.
surrogate_acceleration_proxy = geodesic_acceleration_proxy
surrogate_exp_map_proxy = geodesic_exp_map_proxy


@torch.no_grad()
def log_map_shooting(x, y_target, exp_map_fn: Callable, max_iters=8, lr=0.4, tol=1e-3):
    """Damped endpoint shooting; no global convexity claim is made."""
    base = x.detach()
    target = y_target.detach()
    v = (target - base).clone()
    best_v = v.clone()
    best_loss = torch.full((v.shape[0],), float("inf"), device=v.device, dtype=v.dtype)

    for _ in range(max_iters):
        y_pred = exp_map_fn(base, v)
        residual = target - y_pred
        loss = residual.flatten(1).pow(2).sum(dim=1)
        improved = loss < best_loss
        if improved.any():
            shape = [v.shape[0]] + [1] * (v.ndim - 1)
            best_v = torch.where(improved.view(*shape), v, best_v)
            best_loss = torch.minimum(best_loss, loss)
        if torch.sqrt(loss.mean()).item() < tol:
            break
        v = v + lr * residual
    return best_v.detach()


def log_map_shooting_chunked(x, y, exp_map_fn, max_iters=1, lr=0.5, tol=1e-3, chunk_size=16):
    if chunk_size is None or x.shape[0] <= chunk_size:
        return log_map_shooting(x, y, exp_map_fn, max_iters, lr, tol)
    out = []
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        out.append(log_map_shooting(
            x[start:stop], y[start:stop], exp_map_fn,
            max_iters=max_iters, lr=lr, tol=tol,
        ))
    return torch.cat(out, dim=0)


# -----------------------------------------------------------------------------
# Localized latent measure + mathematically correct constrained Frechet direction
# -----------------------------------------------------------------------------
@torch.no_grad()
def sample_local_proposal_with_importance_weights(
    z_centers,
    n_samples: int,
    sigma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Local proposal:
        q_{z0,sigma}(z) = N(z0, sigma^2 I).

    We define the localized target measure
        nu_{z0,sigma}(dz) ∝ K_sigma(z,z0) mu(dz),
    where mu=N(0,I) and
        K_sigma(z,z0) = exp(-||z-z0||^2/(2 sigma^2)).

    Since q is proportional to K_sigma as a density in z, self-normalized
    importance sampling from q has weights
        w_i ∝ mu(z_i) ∝ exp(-||z_i||^2/2).

    This preserves the finite-sample behavior of the original code while giving
    the weighting a correct importance-sampling interpretation.  The weights do
    NOT depend on the candidate Frechet mean/current optimization variable.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    m = z_centers.shape[0]
    shape = z_centers.shape[1:]

    if n_samples % 2 != 0:
        raise ValueError("n_samples must be even for antithetic sampling")

    half = n_samples // 2
    noise_half = torch.randn(
        m, half, *shape,
        device=z_centers.device,
        dtype=z_centers.dtype,
    )
    noise = torch.cat([noise_half, -noise_half], dim=1)
    Z = z_centers[:, None] + float(sigma) * noise

    # log mu(z_i), constants omitted.
    log_w = -0.5 * Z.flatten(2).pow(2).sum(dim=2)
    weights = torch.softmax(log_w, dim=1)
    return Z, weights


@torch.no_grad()
def sample_localized_measure_direct(
    z_centers,
    n_samples: int,
    sigma: float,
    antithetic: bool = False,
):
    r"""
    Direct sampler for the SAME theoretical localized measure used above:

        nu_{z0,sigma}
          = N(z0/(1+sigma^2), sigma^2/(1+sigma^2) I).

    This function is supplied for validation/ablation only.  The production
    HCC estimator intentionally uses `sample_local_proposal_with_importance_weights`
    because that is the finite-sample estimator used for the earlier successful
    HCC image results.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    m = z_centers.shape[0]
    shape = z_centers.shape[1:]
    denom = 1.0 + float(sigma) ** 2
    mean = z_centers / denom
    std = float(sigma) / math.sqrt(denom)

    if antithetic:
        if n_samples % 2 != 0:
            raise ValueError("n_samples must be even for antithetic sampling")
        half = n_samples // 2
        noise_half = torch.randn(
            m, half, *shape,
            device=z_centers.device,
            dtype=z_centers.dtype,
        )
        noise = torch.cat([noise_half, -noise_half], dim=1)
    else:
        noise = torch.randn(
            m, n_samples, *shape,
            device=z_centers.device,
            dtype=z_centers.dtype,
        )

    Z = mean[:, None] + std * noise
    return Z


@torch.no_grad()
def evaluate_fixed_local_frechet_energy(
    centers_x0,
    candidate_Y,
    weights,
    exp_map_fn: Callable,
    model,
    scheduler,
    proxy_t,
    metric_lambda,
    local_shooting_iters=3,
    local_shooting_lr=0.5,
    pair_chunk_size=16,
):
    r"""
    Evaluate the SAME fixed empirical local Frechet objective used by a line-search
    step:

        F_i(x) = sum_j w_ij d_g(x, Y_ij)^2.

    candidate_Y and weights are held fixed during backtracking.  This is important:
    resampling while testing a step would compare different Monte Carlo objectives.

    Distances are approximated with the same numerical shooting/log routine used to
    construct the descent direction, so the implementation is internally consistent.
    """
    m, n_candidates = weights.shape
    bases = centers_x0[:, None].expand(-1, n_candidates, *centers_x0.shape[1:])
    flat_bases = bases.flatten(0, 1)
    flat_Y = candidate_Y.flatten(0, 1)

    logs = log_map_shooting_chunked(
        flat_bases,
        flat_Y,
        exp_map_fn=exp_map_fn,
        max_iters=local_shooting_iters,
        lr=local_shooting_lr,
        tol=1e-3,
        chunk_size=pair_chunk_size,
    ).view(m, n_candidates, *centers_x0.shape[1:])

    s = score_proxy(centers_x0, model, scheduler, proxy_t).detach()
    s_rep = s[:, None].expand_as(logs)
    euclid_d2 = logs.flatten(2).pow(2).sum(dim=2)
    score_d2 = metric_lambda * (s_rep * logs).flatten(2).sum(dim=2).pow(2)
    energy = (weights * (euclid_d2 + score_d2)).sum(dim=1)
    return energy.detach()


@torch.no_grad()
def local_frechet_direction_clean(
    centers_x0,
    z_centers_T,
    latent_to_clean_fn: Callable,
    exp_map_fn: Callable,
    model,
    scheduler,
    proxy_t,
    metric_lambda,
    sigma=0.2,
    n_candidates=8,
    local_shooting_iters=3,
    local_shooting_lr=0.5,
    pair_chunk_size=16,
):
    r"""
    Compute one importance-sampled local surrogate-Frechet direction in clean
    coordinates.

    The theoretical target is exactly the manuscript localized measure
        nu_i(dz) proportional to K_sigma(z,z_i) mu(dz).
    Numerically, to preserve the earlier successful finite-sample HCC behavior,
    we estimate expectations under nu_i using self-normalized importance
    sampling from q_i=N(z_i,sigma^2 I).

    For each current latent center z_i, draw antithetic proposal samples

        Z_ij ~ q_i = N(z_i, sigma^2 I)

    (marginally; antithetic pairs are dependent but each has the correct q_i
    marginal).  Define the localized target measure

        nu_i(dz) proportional to K_sigma(z,z_i) mu(dz),

    where mu=N(0,I) and K_sigma is the Gaussian localization kernel.  Since q_i
    is proportional to K_sigma as a density in z, self-normalized importance
    weights satisfy

        w_ij proportional to mu(Z_ij) = exp(-||Z_ij||^2/2).

    With Y_ij = Phi(Z_ij), define

        F_i(x) = sum_j w_ij d_g(x,Y_ij)^2.

    In the exact Riemannian construction, the corresponding Fréchet gradient
    uses exact logarithms.  Here those logarithms are replaced by the Eq. (8)
    surrogate shooting directions.  Thus delta_x is a Fréchet-motivated
    numerical direction.  The implementation does not rely on it being the
    exact Riemannian gradient: accepted updates are checked by backtracking on
    the same frozen surrogate objective used to construct the direction.
    """
    m = centers_x0.shape[0]
    Z, weights = sample_local_proposal_with_importance_weights(
        z_centers_T, n_candidates, sigma
    )
    flat_Z = Z.flatten(0, 1)
    flat_Y = latent_to_clean_fn(flat_Z)
    candidate_Y = flat_Y.view(m, n_candidates, *centers_x0.shape[1:])

    bases = centers_x0[:, None].expand(-1, n_candidates, *centers_x0.shape[1:])
    flat_bases = bases.flatten(0, 1)

    logs = log_map_shooting_chunked(
        flat_bases,
        flat_Y,
        exp_map_fn=exp_map_fn,
        max_iters=local_shooting_iters,
        lr=local_shooting_lr,
        tol=1e-3,
        chunk_size=pair_chunk_size,
    ).view(m, n_candidates, *centers_x0.shape[1:])

    wview = weights.view(m, n_candidates, *([1] * (centers_x0.ndim - 1)))
    delta_x = (wview * logs).sum(dim=1)

    s = score_proxy(centers_x0, model, scheduler, proxy_t).detach()
    s_rep = s[:, None].expand_as(logs)
    euclid_d2 = logs.flatten(2).pow(2).sum(dim=2)
    score_d2 = metric_lambda * (s_rep * logs).flatten(2).sum(dim=2).pow(2)
    local_energy = (weights * (euclid_d2 + score_d2)).sum(dim=1)

    candidate_rms = (
        (flat_Y - flat_bases).flatten(1).pow(2).mean(dim=1).sqrt()
        .view(m, n_candidates).mean(dim=1)
    )
    ess = 1.0 / weights.pow(2).sum(dim=1).clamp_min(1e-12)

    return (
        delta_x.detach(),
        local_energy.detach(),
        candidate_rms.detach(),
        ess.detach(),
        s,
        candidate_Y.detach(),
        weights.detach(),
    )


# -----------------------------------------------------------------------------
# Geodesic + generator-constrained clean-space refinement
# -----------------------------------------------------------------------------
@torch.no_grad()
def build_geodesic_curve(xA, xB, log_map_fn, exp_map_fn, n_steps=10):
    """
    Build the ambient Eq. (8) surrogate shooting path.

    This is called `build_geodesic_curve` for backwards compatibility, but in
    Route-2 terminology it is a surrogate geometric path rather than a claim of
    exact Eq. (7) geodesic integration.  The shooting solve is numerical, so the
    surrogate Exp(Log(y)) need not land exactly on y.
    We therefore pin the two boundary values explicitly.  This does not alter the
    interior shooting trajectory and guarantees that the interpolation has the
    requested endpoints.
    """
    v = log_map_fn(xA, xB)
    alphas = torch.linspace(0.0, 1.0, n_steps, device=xA.device, dtype=xA.dtype)
    bases = xA.expand(n_steps, *xA.shape[1:])
    velocities = alphas.view(n_steps, *([1] * (v.ndim - 1))) * v.expand_as(bases)
    curve = exp_map_fn(bases, velocities)
    curve[0:1] = xA
    curve[-1:] = xB
    return curve


def refine_latent_constrained_clean_frechet(
    curve_z_T,
    latent_to_clean_fn: Callable,
    latent_to_clean_grad_fn: Callable,
    exp_map_fn: Callable,
    model,
    scheduler,
    proxy_t,
    metric_lambda,
    sigma=0.2,
    n_iters=10,
    n_candidates=8,
    latent_step_size=0.18,
    step_decay=0.80,
    min_latent_step=0.035,
    local_shooting_iters=3,
    local_shooting_lr=0.5,
    pair_chunk_size=16,
    normalize_step=True,
    backtracking_factor=0.5,
    max_backtracking_steps=4,
    energy_decrease_tol=0.0,
):
    r"""
    Generator-constrained projected local Fréchet-motivated refinement with
    per-node backtracking.

    At outer iteration k, for each interior latent node z_i:

      1. Decode x_i = Phi(z_i).
      2. Draw ONE local Monte Carlo candidate set {Z_ij}, decode Y_ij=Phi(Z_ij),
         and compute fixed normalized importance weights w_ij.
      3. Form the importance-sampled surrogate Fréchet direction

             delta_i = sum_j w_ij Log_{x_i}(Y_ij).

      4. For

             F_i(z) = sum_j w_ij d_g(Phi(z),Y_ij)^2,

         the Euclidean latent gradient is

             grad_z F_i(z_i) = -2 D Phi(z_i)^T g(x_i) delta_i.

         With exact Riemannian logarithms this is the Fréchet descent
         direction.  With the Eq. (8) surrogate logs used here, it is treated as
         a proposal direction and validated numerically by the frozen-objective
         backtracking step below.

      5. Project d_i orthogonally to the discrete latent-curve tangent.  Since
         orthogonal projection suppresses tangential motion.  In the exact
         Fréchet case this preserves descent in the normal subspace; in the
         implemented surrogate case acceptance is determined by backtracking.

      6. Normalize the projected direction and use its scheduled norm as a
         trust-region-like trial step.

      7. Backtrack PER NODE using the SAME {Y_ij,w_ij}.  A trial is accepted only
         when the corresponding fixed empirical Frechet energy does not increase
         (up to energy_decrease_tol).  Otherwise the step is multiplied by
         backtracking_factor.  If no tested step is accepted, that node stays put.

    The neighborhood is refreshed only at the next outer iteration.  No respacing
    is performed.

    Numerical note: Log and d_g are approximated by the same shooting routine in
    both the descent direction and line-search energy, so the implemented
    approximate objective is internally consistent.
    """
    if not (0.0 < backtracking_factor < 1.0):
        raise ValueError("backtracking_factor must lie in (0,1)")
    if max_backtracking_steps < 1:
        raise ValueError("max_backtracking_steps must be >= 1")

    curve_z = curve_z_T.detach().clone()
    history_z = [curve_z.clone()]
    energy_history = []
    accepted_energy_history = []
    step_history = []

    for it in range(n_iters):
        print(f"\nClean-space proxy Frechet refinement {it + 1}/{n_iters}")
        z_centers = curve_z[1:-1].detach()
        m = z_centers.shape[0]

        # One differentiable decoder graph for all current interior nodes.
        with torch.enable_grad():
            z_req = z_centers.clone().requires_grad_(True)
            x_graph = latent_to_clean_grad_fn(z_req)
        centers_x0 = x_graph.detach()

        (
            delta_x,
            local_energy,
            candidate_rms,
            ess,
            s_centers,
            candidate_Y,
            weights,
        ) = local_frechet_direction_clean(
            centers_x0=centers_x0,
            z_centers_T=z_centers,
            latent_to_clean_fn=latent_to_clean_fn,
            exp_map_fn=exp_map_fn,
            model=model,
            scheduler=scheduler,
            proxy_t=proxy_t,
            metric_lambda=metric_lambda,
            sigma=sigma,
            n_candidates=n_candidates,
            local_shooting_iters=local_shooting_iters,
            local_shooting_lr=local_shooting_lr,
            pair_chunk_size=pair_chunk_size,
        )

        # Differential of F_i through Phi.  If grad_g F_i=-2 delta_x, then
        # grad_z F_i=-2 D Phi^T g delta_x; +D Phi^T g delta_x is descent.
        with torch.no_grad():
            pullback_covector = metric_apply_from_score(
                s_centers, delta_x, metric_lambda
            ).detach()

        with torch.enable_grad():
            scalar = (x_graph * pullback_covector).sum()
            latent_dir = torch.autograd.grad(
                scalar, z_req, retain_graph=False, create_graph=False
            )[0].detach()

        # Project in latent space to reduce tangential drift/clumping.
        with torch.no_grad():
            tau_z = curve_z[2:] - curve_z[:-2]
            tau_norm_sq = tau_z.flatten(1).pow(2).sum(dim=1).clamp_min(1e-12)
            dir_dot_tau = (latent_dir * tau_z).flatten(1).sum(dim=1)
            view_shape = [m] + [1] * (latent_dir.ndim - 1)
            latent_dir = latent_dir - (
                dir_dot_tau / tau_norm_sq
            ).view(*view_shape) * tau_z

        dir_norm = latent_dir.flatten(1).norm(dim=1)
        valid_dir = dir_norm > 1e-12
        safe_norm = dir_norm.clamp_min(1e-12)
        view_shape = [m] + [1] * (latent_dir.ndim - 1)

        scheduled_step = max(
            float(min_latent_step),
            float(latent_step_size) * (float(step_decay) ** it),
        )

        if normalize_step:
            unit_dir = latent_dir / safe_norm.view(*view_shape)
        else:
            # In the unnormalized case, unit_dir is simply the raw descent
            # direction and the scalar step multiplies it below.
            unit_dir = latent_dir

        # Per-node monotone backtracking on the FIXED empirical objective.
        accepted_z = z_centers.clone()
        accepted_energy = local_energy.clone()
        accepted = ~valid_dir
        accepted_steps = torch.zeros(m, device=z_centers.device, dtype=z_centers.dtype)
        trial_steps = torch.full(
            (m,), scheduled_step, device=z_centers.device, dtype=z_centers.dtype
        )

        for bt in range(max_backtracking_steps):
            active_idx = torch.nonzero(~accepted, as_tuple=False).flatten()
            if active_idx.numel() == 0:
                break

            step_active = trial_steps[active_idx]
            active_view = [active_idx.numel()] + [1] * (z_centers.ndim - 1)
            trial_z = (
                z_centers[active_idx]
                + step_active.view(*active_view) * unit_dir[active_idx]
            )

            trial_x = latent_to_clean_fn(trial_z)
            trial_energy = evaluate_fixed_local_frechet_energy(
                centers_x0=trial_x,
                candidate_Y=candidate_Y[active_idx],
                weights=weights[active_idx],
                exp_map_fn=exp_map_fn,
                model=model,
                scheduler=scheduler,
                proxy_t=proxy_t,
                metric_lambda=metric_lambda,
                local_shooting_iters=local_shooting_iters,
                local_shooting_lr=local_shooting_lr,
                pair_chunk_size=pair_chunk_size,
            )

            baseline = local_energy[active_idx]
            # energy_decrease_tol is relative.  The default 0 requires monotone
            # decrease; a tiny positive value can be used if numerical shooting
            # noise makes strict monotonicity too brittle.
            threshold = baseline * (1.0 + float(energy_decrease_tol))
            accept_local = trial_energy <= threshold

            if accept_local.any():
                good_idx = active_idx[accept_local]
                accepted_z[good_idx] = trial_z[accept_local]
                accepted_energy[good_idx] = trial_energy[accept_local]
                accepted_steps[good_idx] = step_active[accept_local]
                accepted[good_idx] = True

            reject_local = ~accept_local
            if reject_local.any():
                bad_idx = active_idx[reject_local]
                trial_steps[bad_idx] *= float(backtracking_factor)

        # Nodes that never passed the line search remain unchanged.
        new_curve = curve_z.clone()
        new_curve[1:-1] = accepted_z

        n_accepted = int((accepted_steps > 0).sum().item())
        n_rejected = int(m - n_accepted)
        mean_accepted_step = (
            accepted_steps[accepted_steps > 0].mean().item()
            if n_accepted > 0 else 0.0
        )
        rel_change = (accepted_energy - local_energy) / local_energy.clamp_min(1e-12)

        print(f"scheduled latent step: {scheduled_step:.6g}")
        print(f"accepted nodes: {n_accepted}/{m}; unchanged nodes: {n_rejected}")
        print(f"mean accepted latent step: {mean_accepted_step:.6g}")
        print(f"mean fixed-objective energy before: {local_energy.mean().item():.6g}")
        print(f"mean fixed-objective energy after:  {accepted_energy.mean().item():.6g}")
        print(f"mean relative energy change: {rel_change.mean().item():.6g}")
        print(f"mean candidate clean RMS distance: {candidate_rms.mean().item():.6g}")
        print(f"mean importance ESS: {ess.mean().item():.4f} / {n_candidates}")

        curve_z = new_curve.detach()
        history_z.append(curve_z.clone())
        energy_history.append(local_energy.cpu())
        accepted_energy_history.append(accepted_energy.cpu())
        step_history.append(accepted_steps.cpu())

    return history_z, energy_history, accepted_energy_history, step_history


@torch.no_grad()
def build_linear_latent_curve(zA_T, zB_T, n_steps=10):
    """Simple linear interpolation in latent z_T coordinates."""
    alphas = torch.linspace(0.0, 1.0, n_steps, device=zA_T.device, dtype=zA_T.dtype)
    curve = []
    for a in alphas:
        curve.append((1.0 - a) * zA_T + a * zB_T)
    return torch.cat(curve, dim=0)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def load_image(path, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0)


@torch.no_grad()
def plot_initial_vs_refined(initial_z, refined_z, latent_to_clean_fn, filename):
    """
    Plot the initial latent linear interpolation versus the final refined curve.
    Both rows are decoded through Phi, so the comparison is generator-constrained
    and apples-to-apples.
    """
    pair = torch.cat([initial_z, refined_z], dim=0)
    decoded = latent_to_clean_fn(pair)
    L = initial_z.shape[0]
    initial_x0 = decoded[:L]
    refined_x0 = decoded[L:]
    rows = [initial_x0, refined_x0]

    fig, axes = plt.subplots(2, L, figsize=(1.8 * L, 3.7))
    for i, row in enumerate(rows):
        imgs = (row * 0.5 + 0.5).clamp(0, 1)
        for j in range(L):
            axes[i, j].imshow(transforms.ToPILImage()(imgs[j].cpu()))
            axes[i, j].axis("off")
    fig.suptitle("Latent LERP (top) vs High-Confidence Curve refinement (bottom)")
    plt.tight_layout()
    plt.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_refinement_history(history_z, latent_to_clean_fn, filename, every=1):
    """Save selected refinement iterations so convergence/early stopping is visible."""
    selected = list(range(0, len(history_z), max(1, int(every))))
    if selected[-1] != len(history_z) - 1:
        selected.append(len(history_z) - 1)

    curves = [history_z[i] for i in selected]
    L = curves[0].shape[0]
    decoded = latent_to_clean_fn(torch.cat(curves, dim=0))
    rows = decoded.view(len(curves), L, *decoded.shape[1:])

    fig, axes = plt.subplots(len(curves), L, figsize=(1.8 * L, 1.7 * len(curves)))
    if len(curves) == 1:
        axes = axes[None, :]

    for r, (iter_idx, row) in enumerate(zip(selected, rows)):
        imgs = (row * 0.5 + 0.5).clamp(0, 1)
        for j in range(L):
            axes[r, j].imshow(transforms.ToPILImage()(imgs[j].cpu()))
            axes[r, j].axis("off")
        axes[r, 0].set_ylabel(f"iter {iter_idx}", rotation=0, labelpad=28, va="center")

    #fig.suptitle("Local Frechet refinement history")
    plt.tight_layout()
    plt.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_path_A = "/data5/accounts/marsh/.cache/kagglehub/datasets/andrewmvd/animal-faces/versions/1/afhq/train/dog/flickr_dog_000070.jpg"
    image_path_B = "/data5/accounts/marsh/.cache/kagglehub/datasets/andrewmvd/animal-faces/versions/1/afhq/train/cat/flickr_cat_000070.jpg"
    checkpoint_path = "./vp_diffusion_outputs/unet_animal_epoch_2000.pt"

    # ------------------------------------------------------------------
    # Paper parameters
    # ------------------------------------------------------------------
    proxy_t = 400            # ONLY a score proxy; geometry remains in x_0 coordinates.
    metric_lambda = 1e6

    exp_ode_steps = 6
    global_shooting_iters = 24
    global_shooting_lr = 0.25
    local_shooting_iters = 3
    local_shooting_lr = 0.5
    n_geo_points = 10

    # Numerical DDIM Phi used for generator-constrained refinement.
    # Increase to 64/100 for final figures if runtime allows.
    # Restore the high-fidelity DDIM settings used by the original script.
    # The previous fast version used 100 inversion steps and only 40 decode
    # steps, which materially degraded interpolation image quality.
    inversion_steps = 1000
    refine_decode_steps = 100   # geometry/refinement; increase toward 1000 for fidelity
    display_decode_steps = 1000 # final paper-quality rendering

    sigma = 0.20
    n_candidates = 16
    refinement_iters = 50
    latent_step_size = 0.18
    step_decay = 0.80
    min_latent_step = 0.035
    pair_chunk_size = 16

    # Per-node monotone backtracking prevents isolated overshoot artifacts.
    backtracking_factor = 0.5
    max_backtracking_steps = 4
    energy_decrease_tol = 0.0

    print(f"device={device}")
    print(
        "localization estimator=self-normalized importance sampling "
        "(same theoretical nu as the manuscript; preserves earlier finite-N behavior)"
    )
    print(
        f"proxy_t={proxy_t}, lambda={metric_lambda:g}, sigma={sigma:g}, "
        f"candidates={n_candidates}, refinement_iters={refinement_iters}, "
        f"latent_step0={latent_step_size:g}, decay={step_decay:g}, "
        f"min_step={min_latent_step:g}"
    )

    x0_A = load_image(image_path_A).to(device)
    x0_B = load_image(image_path_B).to(device)
    x0_pair = torch.cat([x0_A, x0_B], dim=0)

    model = UNetSD().to(device)
    scheduler = VPScheduler(num_timesteps=1000)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    T = scheduler.num_timesteps - 1

    print("Inverting endpoints to z_T...")
    z_pair_T = ddim_forward_inversion_segment(
        model, scheduler, x0_pair, 0, T, num_steps=inversion_steps
    )

    @torch.no_grad()
    def latent_to_clean_fn(z_T):
        """Decoder used inside refinement."""
        return ddim_reverse_segment(
            model, scheduler, z_T, T, 0, num_steps=refine_decode_steps
        )

    @torch.no_grad()
    def latent_to_clean_display_fn(z_T):
        """High-fidelity decoder used only for final figures."""
        return ddim_reverse_segment(
            model, scheduler, z_T, T, 0, num_steps=display_decode_steps
        )

    def latent_to_clean_grad_fn(z_T):
        return ddim_reverse_segment_differentiable(
            model, scheduler, z_T, T, 0,
            num_steps=refine_decode_steps, use_checkpoint=True,
        )

    @torch.no_grad()
    def clean_to_latent_fn(x0):
        return ddim_forward_inversion_segment(
            model, scheduler, x0, 0, T, num_steps=inversion_steps
        )

    # Use reconstructed generated endpoints so the top and bottom curves share
    # endpoints belonging to the same numerical DDIM map Phi.
    x_pair_gen = latent_to_clean_display_fn(z_pair_T)
    xA_gen, xB_gen = x_pair_gen[0:1], x_pair_gen[1:2]

    def exp_map_fn(x, v):
        return geodesic_exp_map_proxy(
            x, v,
            model=model,
            scheduler=scheduler,
            proxy_t=proxy_t,
            metric_lambda=metric_lambda,
            n_steps=exp_ode_steps,
        )

    def global_log_map_fn(x, y):
        return log_map_shooting(
            x, y, exp_map_fn,
            max_iters=global_shooting_iters,
            lr=global_shooting_lr,
            tol=1e-3,
        )

    print("Building initial LINEAR interpolation in latent z_T coordinates...")
    curve_z0 = build_linear_latent_curve(
        z_pair_T[0:1], z_pair_T[1:2], n_steps=n_geo_points
    )

    print("Running generator-constrained clean-space local Frechet refinement from latent LERP...")
    history_z, energy_history, accepted_energy_history, step_history = refine_latent_constrained_clean_frechet(
        curve_z0,
        latent_to_clean_fn=latent_to_clean_fn,
        latent_to_clean_grad_fn=latent_to_clean_grad_fn,
        exp_map_fn=exp_map_fn,
        model=model,
        scheduler=scheduler,
        proxy_t=proxy_t,
        metric_lambda=metric_lambda,
        sigma=sigma,
        n_iters=refinement_iters,
        n_candidates=n_candidates,
        latent_step_size=latent_step_size,
        step_decay=step_decay,
        min_latent_step=min_latent_step,
        local_shooting_iters=local_shooting_iters,
        local_shooting_lr=local_shooting_lr,
        pair_chunk_size=pair_chunk_size,
        normalize_step=True,
        backtracking_factor=backtracking_factor,
        max_backtracking_steps=max_backtracking_steps,
        energy_decrease_tol=energy_decrease_tol,
    )

    out_name = "Corrected_Fast_Frechet_LINEAR_INIT_BACKTRACK.png"
    plot_initial_vs_refined(
        curve_z0, history_z[-1], latent_to_clean_display_fn, out_name
    )
    print(f"Saved {out_name}")

    history_name = "Corrected_Fast_Frechet_HISTORY_BACKTRACK.png"
    plot_refinement_history(
        history_z, latent_to_clean_display_fn, history_name, every=1
    )
    print(f"Saved {history_name}")

    if energy_history:
        print("Final pre-step local Frechet energies:", energy_history[-1].tolist())
    if accepted_energy_history:
        print("Final accepted local Frechet energies:", accepted_energy_history[-1].tolist())
    if step_history:
        print("Final accepted per-node step sizes:", step_history[-1].tolist())


if __name__ == "__main__":
    main()
