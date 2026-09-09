#!/usr/bin/env python3
"""Score-force NEB-style baseline for the interpolation experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

import High_Confidence_Curves as hcc
from experiment_utils import (
    choose_device,
    decode_display,
    invert_pair,
    load_config,
    save_curve_grid,
    save_json,
    seed_everything,
    strict_load_model,
)


def reparameterize_curve(curve: torch.Tensor) -> torch.Tensor:
    """Redistribute nodes approximately uniformly in Euclidean latent arclength."""
    flat = curve.flatten(1)
    seg = (flat[1:] - flat[:-1]).norm(dim=1)
    cum = torch.cat([torch.zeros(1, device=curve.device, dtype=curve.dtype), seg.cumsum(0)])
    total = cum[-1]
    if float(total.item()) <= 1e-12:
        return curve.clone()
    targets = torch.linspace(0, float(total.item()), curve.shape[0], device=curve.device, dtype=curve.dtype)
    out = [curve[0:1]]
    for t in targets[1:-1]:
        j = int(torch.searchsorted(cum, t, right=True).item()) - 1
        j = max(0, min(j, curve.shape[0] - 2))
        denom = (cum[j + 1] - cum[j]).clamp_min(1e-12)
        w = (t - cum[j]) / denom
        out.append(((1 - w) * curve[j:j+1] + w * curve[j+1:j+2]))
    out.append(curve[-1:])
    return torch.cat(out, dim=0)


@torch.no_grad()
def score_force_neb(
    curve: torch.Tensor,
    model,
    scheduler,
    *,
    proxy_t: int,
    n_iters: int,
    lr: float,
    spring_k: float,
    reparam_every: int = 5,
    convergence_tol: float = 0.0,
):
    """NEB-style score-force refinement with fixed endpoints.

    Returns ``(final_curve, diagnostics)``.  If ``convergence_tol > 0``, the run
    terminates when the per-coordinate RMS node update falls below that value.
    """
    curve = curve.clone()
    history_delta = []
    converged = False

    for it in range(int(n_iters)):
        new_curve = curve.clone()
        for i in range(1, curve.shape[0] - 1):
            x_i = curve[i]
            prev = curve[i - 1]
            nxt = curve[i + 1]
            tau = (nxt - prev).flatten()
            tau = tau / tau.norm().clamp_min(1e-12)

            score = hcc.score_proxy(
                x_i.unsqueeze(0), model, scheduler, int(proxy_t)
            ).flatten()
            score_perp = score - torch.dot(score, tau) * tau

            d_prev = (x_i - prev).flatten().norm()
            d_next = (nxt - x_i).flatten().norm()
            spring = -float(spring_k) * (d_prev - d_next) * tau
            dx = float(lr) * (score_perp + spring)
            new_curve[i] = curve[i] + dx.view_as(x_i)

        new_curve[0] = curve[0]
        new_curve[-1] = curve[-1]
        if reparam_every > 0 and (it + 1) % int(reparam_every) == 0:
            new_curve = reparameterize_curve(new_curve)
            new_curve[0] = curve[0]
            new_curve[-1] = curve[-1]

        delta = (new_curve - curve).pow(2).mean().sqrt().item()
        history_delta.append(float(delta))
        curve = new_curve

        if convergence_tol > 0 and delta <= convergence_tol:
            converged = True
            break

    return curve, {
        "iterations_run": len(history_delta),
        "converged": converged,
        "final_update_rmse": history_delta[-1] if history_delta else 0.0,
        "update_rmse_history": history_delta,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Score-force NEB-style parameter sweep")
    p.add_argument("--image-a", required=True)
    p.add_argument("--image-b", required=True)
    p.add_argument("--config", default="configs/afhq_paper.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output-dir", default="outputs/neb_sweep")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-iters", type=int, default=None)
    p.add_argument("--proxy-t", type=int, default=None)
    p.add_argument("--lr-values", nargs="+", type=float, default=None)
    p.add_argument("--k-values", nargs="+", type=float, default=None)
    p.add_argument("--reparam-every", type=int, default=None)
    p.add_argument("--convergence-tol", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    neb_cfg = cfg.get("neb", {})
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    seed_everything(seed)
    device = choose_device(args.device)
    checkpoint = args.checkpoint or cfg.get("checkpoint")
    if checkpoint is None:
        raise ValueError("No checkpoint provided")

    n_iters = int(neb_cfg.get("n_iters", 500) if args.n_iters is None else args.n_iters)
    proxy_t = int(neb_cfg.get("proxy_t", cfg["proxy_t"]) if args.proxy_t is None else args.proxy_t)
    lr_values = args.lr_values or list(neb_cfg.get("lr_values", [1e-3, 1e-2]))
    k_values = args.k_values or list(neb_cfg.get("k_values", [0.0, 1.0]))
    reparam_every = int(neb_cfg.get("reparam_every", 5) if args.reparam_every is None else args.reparam_every)
    tol = float(neb_cfg.get("convergence_tol", 0.0) if args.convergence_tol is None else args.convergence_tol)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, scheduler = strict_load_model(checkpoint, device, int(cfg.get("num_timesteps", 1000)))

    image_size = int(cfg.get("image_size", 64))
    x_pair = torch.cat([
        hcc.load_image(args.image_a, image_size=image_size),
        hcc.load_image(args.image_b, image_size=image_size),
    ], dim=0).to(device)
    z_pair = invert_pair(model, scheduler, x_pair, cfg)
    initial_z = hcc.build_linear_latent_curve(
        z_pair[0:1], z_pair[1:2], n_steps=int(neb_cfg.get("n_curve_points", cfg["n_geo_points"]))
    )

    conditions = []
    json_conditions = {}
    for lr in lr_values:
        for k in k_values:
            print(f"Running score-force NEB: lr={lr:g}, k={k:g}, max_iters={n_iters}")
            seed_everything(seed)
            final_z, diag = score_force_neb(
                initial_z,
                model,
                scheduler,
                proxy_t=proxy_t,
                n_iters=n_iters,
                lr=float(lr),
                spring_k=float(k),
                reparam_every=reparam_every,
                convergence_tol=tol,
            )
            final_x = decode_display(model, scheduler, final_z, cfg)
            conditions.append((f"lr={lr:g}, k={k:g}", final_x))
            json_conditions[f"lr={lr:g},k={k:g}"] = diag

    # Large sweeps are split into manageable figures rather than one unreadable mosaic.
    chunk = 8
    for j in range(0, len(conditions), chunk):
        save_curve_grid(
            conditions[j:j+chunk],
            out_dir / f"neb_sweep_{j//chunk:03d}.png",
            dpi=180,
        )

    save_json({
        "baseline_name": "score-force NEB-style heuristic",
        "caveat": (
            "The learned score proxy is used directly as a force and is not assumed "
            "to be the gradient of a scalar potential."
        ),
        "image_a": str(Path(args.image_a).resolve()),
        "image_b": str(Path(args.image_b).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "seed": seed,
        "proxy_t": proxy_t,
        "max_iterations": n_iters,
        "reparameterize_every": reparam_every,
        "convergence_tol": tol,
        "lr_values": [float(x) for x in lr_values],
        "k_values": [float(x) for x in k_values],
        "conditions": json_conditions,
    }, out_dir / "neb_sweep.json")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
