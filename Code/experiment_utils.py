#!/usr/bin/env python3
"""Shared utilities for the paper-facing HCC experiments.

This module deliberately reuses the production implementation in
``High_Confidence_Curves.py`` rather than reimplementing the model, scheduler,
DDIM maps, or HCC update in multiple scripts.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torchvision import transforms

# Self-contained same-folder import: place this file beside High_Confidence_Curves.py.
CODE_DIR = Path(__file__).resolve().parent

import High_Confidence_Curves as hcc


def resolve_path(path: os.PathLike | str, *, must_exist: bool = False) -> Path:
    """Resolve paths robustly when all paper scripts live in one Code/ folder.

    Search order includes the current working directory, this Code directory, and
    several ancestor directories.  This lets both of these layouts work:

      repo/Code/<scripts>
      repo/Diffusion_T2_upgrade/Code/<scripts>

    It also treats ``configs/afhq_paper.yaml`` as a request that may be satisfied
    by a same-folder ``afhq_paper.yaml``.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        out = p.resolve()
        if must_exist and not out.exists():
            raise FileNotFoundError(out)
        return out

    candidates = []
    # What the user typed relative to where they launched Python.
    candidates.append((Path.cwd() / p).resolve())
    # Relative to this Code folder.
    candidates.append((CODE_DIR / p).resolve())
    # If the default says configs/foo.yaml but foo.yaml is beside the scripts.
    candidates.append((CODE_DIR / p.name).resolve())

    # Try ancestors so repo-relative things like Model/... work even when this
    # Code folder is nested inside a temporary upgrade folder.
    for anc in [CODE_DIR.parent, *list(CODE_DIR.parents)[:5]]:
        candidates.append((anc / p).resolve())
        candidates.append((anc / p.name).resolve())

    # Preserve order while removing duplicates.
    seen = set()
    unique = []
    for c in candidates:
        cs = str(c)
        if cs not in seen:
            seen.add(cs)
            unique.append(c)

    for c in unique:
        if c.exists():
            return c

    if must_exist:
        tried = "\n  - ".join(str(c) for c in unique)
        raise FileNotFoundError(f"Could not resolve {path!r}. Tried:\n  - {tried}")

    # For outputs / not-yet-created paths, prefer CWD-relative semantics.
    return unique[0]


def choose_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {requested!r}, but CUDA is unavailable")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: os.PathLike | str) -> Dict:
    path = resolve_path(path, must_exist=True)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Configuration in {path} must be a YAML mapping")
    return data


def resolve_repo_path(path: str | os.PathLike, repo_root: Optional[os.PathLike] = None) -> Path:
    if repo_root is not None:
        p = Path(path).expanduser()
        return p.resolve() if p.is_absolute() else (Path(repo_root) / p).resolve()
    return resolve_path(path)


def strict_load_model(
    checkpoint: str | os.PathLike,
    device: torch.device,
    num_timesteps: int = 1000,
):
    """Load the checkpoint or fail loudly; never fall back to random weights."""
    checkpoint = resolve_path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Refusing to run with random weights."
        )

    model = hcc.UNetSD().to(device)
    scheduler = hcc.VPScheduler(num_timesteps=num_timesteps)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, scheduler


@torch.no_grad()
def invert_pair(model, scheduler, x_pair: torch.Tensor, cfg: Dict) -> torch.Tensor:
    T = scheduler.num_timesteps - 1
    return hcc.ddim_forward_inversion_segment(
        model,
        scheduler,
        x_pair,
        0,
        T,
        num_steps=int(cfg["inversion_steps"]),
    )


@torch.no_grad()
def decode_display(model, scheduler, z_T: torch.Tensor, cfg: Dict) -> torch.Tensor:
    T = scheduler.num_timesteps - 1
    return hcc.ddim_reverse_segment(
        model,
        scheduler,
        z_T,
        T,
        0,
        num_steps=int(cfg["display_decode_steps"]),
    )


def make_hcc_maps(model, scheduler, cfg: Dict, *, proxy_t: Optional[int] = None,
                  metric_lambda: Optional[float] = None):
    T = scheduler.num_timesteps - 1
    proxy_t = int(cfg["proxy_t"] if proxy_t is None else proxy_t)
    metric_lambda = float(cfg["metric_lambda"] if metric_lambda is None else metric_lambda)

    @torch.no_grad()
    def latent_to_clean_fn(z_T):
        return hcc.ddim_reverse_segment(
            model,
            scheduler,
            z_T,
            T,
            0,
            num_steps=int(cfg["refine_decode_steps"]),
        )

    def latent_to_clean_grad_fn(z_T):
        return hcc.ddim_reverse_segment_differentiable(
            model,
            scheduler,
            z_T,
            T,
            0,
            num_steps=int(cfg["refine_decode_steps"]),
            use_checkpoint=True,
        )

    def exp_map_fn(x, v):
        return hcc.geodesic_exp_map_proxy(
            x,
            v,
            model=model,
            scheduler=scheduler,
            proxy_t=proxy_t,
            metric_lambda=metric_lambda,
            n_steps=int(cfg["exp_ode_steps"]),
        )

    return latent_to_clean_fn, latent_to_clean_grad_fn, exp_map_fn


def run_hcc_from_latents(
    model,
    scheduler,
    z_pair_T: torch.Tensor,
    cfg: Dict,
    *,
    seed: int,
    proxy_t: Optional[int] = None,
    metric_lambda: Optional[float] = None,
):
    """Run one HCC condition from a pair of already inverted endpoints."""
    seed_everything(seed)
    proxy_t = int(cfg["proxy_t"] if proxy_t is None else proxy_t)
    metric_lambda = float(cfg["metric_lambda"] if metric_lambda is None else metric_lambda)
    latent_to_clean_fn, latent_to_clean_grad_fn, exp_map_fn = make_hcc_maps(
        model, scheduler, cfg, proxy_t=proxy_t, metric_lambda=metric_lambda
    )
    curve_z0 = hcc.build_linear_latent_curve(
        z_pair_T[0:1], z_pair_T[1:2], n_steps=int(cfg["n_geo_points"])
    )
    history_z, energy_history, accepted_energy_history, step_history = (
        hcc.refine_latent_constrained_clean_frechet(
            curve_z0,
            latent_to_clean_fn=latent_to_clean_fn,
            latent_to_clean_grad_fn=latent_to_clean_grad_fn,
            exp_map_fn=exp_map_fn,
            model=model,
            scheduler=scheduler,
            proxy_t=proxy_t,
            metric_lambda=metric_lambda,
            sigma=float(cfg["sigma"]),
            n_iters=int(cfg["refinement_iters"]),
            n_candidates=int(cfg["n_candidates"]),
            latent_step_size=float(cfg["latent_step_size"]),
            step_decay=float(cfg["step_decay"]),
            min_latent_step=float(cfg["min_latent_step"]),
            local_shooting_iters=int(cfg["local_shooting_iters"]),
            local_shooting_lr=float(cfg["local_shooting_lr"]),
            pair_chunk_size=int(cfg["pair_chunk_size"]),
            normalize_step=bool(cfg.get("normalize_step", True)),
            backtracking_factor=float(cfg["backtracking_factor"]),
            max_backtracking_steps=int(cfg["max_backtracking_steps"]),
            energy_decrease_tol=float(cfg["energy_decrease_tol"]),
        )
    )
    return {
        "initial_z": curve_z0.detach(),
        "final_z": history_z[-1].detach(),
        "history_z": history_z,
        "energy_history": energy_history,
        "accepted_energy_history": accepted_energy_history,
        "step_history": step_history,
        "proxy_t": proxy_t,
        "metric_lambda": metric_lambda,
    }


def mean_history(xs: Sequence[torch.Tensor]) -> List[float]:
    return [float(x.detach().double().mean().item()) for x in xs]


def summarize_step_history(step_history: Sequence[torch.Tensor]) -> Dict:
    if not step_history:
        return {"acceptance_fraction": None, "mean_accepted_step": None}
    flat = torch.cat([x.detach().float().reshape(-1) for x in step_history])
    accepted = flat > 0
    return {
        "acceptance_fraction": float(accepted.float().mean().item()),
        "mean_accepted_step": float(flat[accepted].mean().item()) if accepted.any() else 0.0,
    }


@torch.no_grad()
def final_importance_ess(final_z: torch.Tensor, cfg: Dict, seed: int) -> Dict:
    """ESS diagnostic at the final interior nodes using a fresh deterministic draw."""
    if final_z.shape[0] <= 2:
        return {"mean": None, "min": None, "max": None}
    seed_everything(seed)
    _, w = hcc.sample_local_proposal_with_importance_weights(
        final_z[1:-1], int(cfg["n_candidates"]), float(cfg["sigma"])
    )
    ess = 1.0 / w.pow(2).sum(dim=1).clamp_min(1e-30)
    return {
        "mean": float(ess.mean().item()),
        "min": float(ess.min().item()),
        "max": float(ess.max().item()),
    }


def slerp_curve(z_a: torch.Tensor, z_b: torch.Tensor, n_steps: int, eps: float = 1e-7):
    """Stable spherical interpolation of two batched latent tensors."""
    if z_a.shape != z_b.shape or z_a.shape[0] != 1:
        raise ValueError("z_a and z_b must have matching shape with batch size 1")
    a = z_a.flatten(1)
    b = z_b.flatten(1)
    a_norm = a / a.norm(dim=1, keepdim=True).clamp_min(eps)
    b_norm = b / b.norm(dim=1, keepdim=True).clamp_min(eps)
    dot = (a_norm * b_norm).sum(dim=1).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    out = []
    for t in torch.linspace(0.0, 1.0, n_steps, device=z_a.device, dtype=z_a.dtype):
        if float(sin_omega.abs().max().item()) < 1e-5:
            z = (1 - t) * z_a + t * z_b
        else:
            c0 = torch.sin((1 - t) * omega) / sin_omega
            c1 = torch.sin(t * omega) / sin_omega
            z = (c0[:, None] * a + c1[:, None] * b).view_as(z_a)
        out.append(z)
    return torch.cat(out, dim=0)


def discover_images(root: str | os.PathLike, class_name: Optional[str] = None) -> List[Path]:
    root = Path(root)
    if class_name:
        candidates = [root / class_name, root / "train" / class_name]
        roots = [p for p in candidates if p.exists()]
        if not roots:
            roots = [root]
    else:
        roots = [root]
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files: List[Path] = []
    for r in roots:
        files.extend(p for p in r.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No images found under {root} (class={class_name!r})")
    return files


def to_unit_interval(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0, 1)


@torch.no_grad()
def save_curve_grid(rows: Sequence[Tuple[str, torch.Tensor]], path: os.PathLike | str,
                    dpi: int = 220) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("rows cannot be empty")
    nrows = len(rows)
    ncols = rows[0][1].shape[0]
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.55 * ncols, 1.6 * nrows))
    if nrows == 1:
        axes = np.asarray(axes)[None, :]
    elif ncols == 1:
        axes = np.asarray(axes)[:, None]
    for r, (label, imgs) in enumerate(rows):
        imgs = to_unit_interval(imgs.detach().cpu())
        for c in range(ncols):
            axes[r, c].imshow(transforms.ToPILImage()(imgs[c]))
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(label, rotation=0, labelpad=48, va="center")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_json(obj: Dict, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def bootstrap_mean_ci(values: Sequence[float], seed: int = 0, n_boot: int = 2000,
                      alpha: float = 0.05) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "low": float(np.quantile(means, alpha / 2)),
        "high": float(np.quantile(means, 1 - alpha / 2)),
    }
