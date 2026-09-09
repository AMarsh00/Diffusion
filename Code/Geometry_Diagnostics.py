#!/usr/bin/env python3
"""Numerical validation for the Eq. (8) image-scale surrogate.

Reports:
  * shooting endpoint RMSE and success rate for several ODE resolutions;
  * fixed-tangent integration sensitivity relative to the finest resolution;
  * Jacobian asymmetry diagnostic ||Jv-J^Tv||/(||Jv||+||J^Tv||);
  * norm of the omitted antisymmetric acceleration term relative to the full and
    retained accelerations from paper Eq. (7).

The diagnostic intentionally does NOT claim that Eq. (8) equals Eq. (7). It
quantifies how large the omitted term is on sampled local trajectories.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import numpy as np
import torch

import High_Confidence_Curves as hcc
from experiment_utils import (
    choose_device,
    load_config,
    save_json,
    seed_everything,
    strict_load_model,
)


def batch_dot(a, b):
    return (a * b).flatten(1).sum(dim=1)


def metric_inverse_apply(s, w, lam: float):
    # Sherman-Morrison for (I + lam ss^T)^(-1) w.
    sw = batch_dot(s, w)
    ss = batch_dot(s, s)
    shape = [s.shape[0]] + [1] * (s.ndim - 1)
    return w - (float(lam) * sw / (1.0 + float(lam) * ss)).view(*shape) * s


def jvp_and_jtv(x, v, model, scheduler, proxy_t):
    s, jv = hcc.score_proxy_and_jvp(x, v, model, scheduler, int(proxy_t))
    with torch.enable_grad():
        x_req = x.detach().requires_grad_(True)
        s_req = hcc.score_proxy(x_req, model, scheduler, int(proxy_t))
        scalar = (s_req * v.detach()).sum()
        jtv = torch.autograd.grad(scalar, x_req, create_graph=False)[0]
    return s.detach(), jv.detach(), jtv.detach()


def acceleration_diagnostics(x, v, model, scheduler, proxy_t, lam):
    s, jv, jtv = jvp_and_jtv(x, v, model, scheduler, proxy_t)
    vjv = batch_dot(v, jv)
    sv = batch_dot(s, v)
    shape = [x.shape[0]] + [1] * (x.ndim - 1)

    retained_bracket = s * vjv.view(*shape)
    omitted_bracket = sv.view(*shape) * (jv - jtv)
    retained = -float(lam) * metric_inverse_apply(s, retained_bracket, lam)
    omitted = -float(lam) * metric_inverse_apply(s, omitted_bracket, lam)
    full = retained + omitted

    def norm(z):
        return z.flatten(1).norm(dim=1)

    jv_n = norm(jv)
    jtv_n = norm(jtv)
    asym = norm(jv - jtv) / (jv_n + jtv_n).clamp_min(1e-12)
    retained_n = norm(retained)
    omitted_n = norm(omitted)
    full_n = norm(full)
    return {
        "jacobian_asymmetry_rel": float(asym.mean().item()),
        "omitted_over_retained": float((omitted_n / retained_n.clamp_min(1e-12)).mean().item()),
        "omitted_over_full": float((omitted_n / full_n.clamp_min(1e-12)).mean().item()),
        "retained_accel_norm": float(retained_n.mean().item()),
        "omitted_accel_norm": float(omitted_n.mean().item()),
        "full_accel_norm": float(full_n.mean().item()),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Validate Eq. (8) surrogate numerics")
    p.add_argument("--config", default="configs/afhq_paper.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output-dir", default="outputs/geometry_diagnostics")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-pairs", type=int, default=None)
    p.add_argument("--exp-step-values", nargs="+", type=int, default=None)
    p.add_argument("--shooting-iters", type=int, default=None)
    p.add_argument("--shooting-lr", type=float, default=None)
    p.add_argument("--shooting-tol", type=float, default=None)
    p.add_argument("--local-latent-sigma", type=float, default=0.05)
    return p.parse_args()


@torch.no_grad()
def decode(model, scheduler, z, cfg):
    T = scheduler.num_timesteps - 1
    return hcc.ddim_reverse_segment(
        model, scheduler, z, T, 0, num_steps=int(cfg["refine_decode_steps"])
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dcfg = cfg.get("diagnostics", {})
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    seed_everything(seed)
    device = choose_device(args.device)
    checkpoint = args.checkpoint or cfg.get("checkpoint")
    if checkpoint is None:
        raise ValueError("No checkpoint provided")
    model, scheduler = strict_load_model(checkpoint, device, int(cfg.get("num_timesteps", 1000)))

    n_pairs = int(dcfg.get("num_pairs", 16) if args.num_pairs is None else args.num_pairs)
    step_values = args.exp_step_values or list(dcfg.get("exp_step_values", [6, 12, 24, 48]))
    shooting_iters = int(dcfg.get("shooting_iters", 8) if args.shooting_iters is None else args.shooting_iters)
    shooting_lr = float(dcfg.get("shooting_lr", 0.4) if args.shooting_lr is None else args.shooting_lr)
    shooting_tol = float(dcfg.get("shooting_tol", 1e-3) if args.shooting_tol is None else args.shooting_tol)
    proxy_t = int(cfg["proxy_t"])
    lam = float(cfg["metric_lambda"])
    image_size = int(cfg.get("image_size", 64))

    rows = []
    fine_steps = max(step_values)
    for pair_idx in range(n_pairs):
        z0 = torch.randn(1, 3, image_size, image_size, device=device)
        z1 = z0 + float(args.local_latent_sigma) * torch.randn_like(z0)
        x0 = decode(model, scheduler, z0, cfg)
        x1 = decode(model, scheduler, z1, cfg)

        pair_row: Dict = {"pair": pair_idx}
        logs = {}
        residuals = {}
        for steps in step_values:
            def exp_map(x, v, steps=steps):
                return hcc.geodesic_exp_map_proxy(
                    x, v, model=model, scheduler=scheduler, proxy_t=proxy_t,
                    metric_lambda=lam, n_steps=int(steps)
                )
            v = hcc.log_map_shooting(
                x0, x1, exp_map, max_iters=shooting_iters,
                lr=shooting_lr, tol=shooting_tol
            )
            endpoint = exp_map(x0, v)
            rmse = (endpoint - x1).flatten(1).pow(2).mean(dim=1).sqrt().item()
            l2 = (endpoint - x1).flatten(1).norm(dim=1).item()
            logs[int(steps)] = v
            residuals[int(steps)] = {"rmse": float(rmse), "l2": float(l2)}
            pair_row[f"shoot_rmse_S{steps}"] = float(rmse)

        v_fine = logs[fine_steps]
        fine_endpoint = hcc.geodesic_exp_map_proxy(
            x0, v_fine, model=model, scheduler=scheduler, proxy_t=proxy_t,
            metric_lambda=lam, n_steps=fine_steps
        )
        for steps in step_values:
            endpoint = hcc.geodesic_exp_map_proxy(
                x0, v_fine, model=model, scheduler=scheduler, proxy_t=proxy_t,
                metric_lambda=lam, n_steps=int(steps)
            )
            sens = (endpoint - fine_endpoint).flatten(1).pow(2).mean(dim=1).sqrt().item()
            pair_row[f"fixed_v_sensitivity_S{steps}_vs_S{fine_steps}"] = float(sens)

        pair_row.update(acceleration_diagnostics(x0, v_fine, model, scheduler, proxy_t, lam))
        rows.append(pair_row)
        print(
            f"[{pair_idx+1}/{n_pairs}] fine RMSE={pair_row[f'shoot_rmse_S{fine_steps}']:.4g}, "
            f"asym={pair_row['jacobian_asymmetry_rel']:.4g}, "
            f"omitted/full={pair_row['omitted_over_full']:.4g}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "geometry_diagnostics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "seed": seed,
        "num_pairs": n_pairs,
        "proxy_t": proxy_t,
        "metric_lambda": lam,
        "exp_step_values": step_values,
        "shooting_iters": shooting_iters,
        "shooting_lr": shooting_lr,
        "shooting_tol": shooting_tol,
        "local_latent_sigma": args.local_latent_sigma,
        "means": {},
        "success_fraction": {},
    }
    numeric_keys = [k for k in rows[0].keys() if k != "pair"]
    for key in numeric_keys:
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        summary["means"][key] = float(vals.mean())
    for steps in step_values:
        vals = np.asarray([r[f"shoot_rmse_S{steps}"] for r in rows])
        summary["success_fraction"][str(steps)] = float((vals <= shooting_tol).mean())

    save_json(summary, out_dir / "geometry_diagnostics_summary.json")
    print(f"Saved {csv_path}")
    print(f"Saved {out_dir / 'geometry_diagnostics_summary.json'}")


if __name__ == "__main__":
    main()
