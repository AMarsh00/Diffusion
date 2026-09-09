#!/usr/bin/env python3
"""Paper lambda ablation for High-Confidence Curves.

The only intended change between conditions is ``metric_lambda``.  The RNG is
reset before every condition so that all lambda values receive the same local
proposal-noise sequence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import High_Confidence_Curves as hcc
from experiment_utils import (
    choose_device,
    decode_display,
    invert_pair,
    load_config,
    mean_history,
    run_hcc_from_latents,
    save_curve_grid,
    save_json,
    seed_everything,
    strict_load_model,
)


def parse_args():
    p = argparse.ArgumentParser(description="HCC lambda ablation")
    p.add_argument("--image-a", required=True)
    p.add_argument("--image-b", required=True)
    p.add_argument("--config", default="configs/afhq_paper.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output-dir", default="outputs/lambda_ablation")
    p.add_argument("--lambda-values", nargs="+", type=float,
                   default=[0.0, 1.0e4, 1.0e5, 1.0e6])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    seed_everything(seed)
    device = choose_device(args.device)
    checkpoint = args.checkpoint or cfg.get("checkpoint")
    if checkpoint is None:
        raise ValueError("No checkpoint provided")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, scheduler = strict_load_model(checkpoint, device, int(cfg.get("num_timesteps", 1000)))

    image_size = int(cfg.get("image_size", 64))
    x_pair = torch.cat([
        hcc.load_image(args.image_a, image_size=image_size),
        hcc.load_image(args.image_b, image_size=image_size),
    ], dim=0).to(device)
    z_pair = invert_pair(model, scheduler, x_pair, cfg)

    lerp_z = hcc.build_linear_latent_curve(
        z_pair[0:1], z_pair[1:2], n_steps=int(cfg["n_geo_points"])
    )
    lerp_x = decode_display(model, scheduler, lerp_z, cfg)

    rows = [("Latent LERP", lerp_x)]
    results = {
        "image_a": str(Path(args.image_a).resolve()),
        "image_b": str(Path(args.image_b).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "seed": seed,
        "parameters": cfg,
        "conditions": {},
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for lam in args.lambda_values:
        print(f"Running lambda={lam:g}")
        run = run_hcc_from_latents(
            model, scheduler, z_pair, cfg, seed=seed, metric_lambda=float(lam)
        )
        final_x = decode_display(model, scheduler, run["final_z"], cfg)
        rows.append((rf"$\lambda={lam:g}$", final_x))
        pre = mean_history(run["energy_history"])
        accepted = mean_history(run["accepted_energy_history"])
        results["conditions"][str(lam)] = {
            "mean_pre_objective_history": pre,
            "mean_accepted_objective_history": accepted,
            "final_node_steps": run["step_history"][-1].tolist() if run["step_history"] else [],
        }
        if accepted:
            ax.plot(range(1, len(accepted) + 1), accepted, label=rf"$\lambda={lam:g}$")

    save_curve_grid(rows, out_dir / "lambda_ablation.png")
    ax.set_xlabel("HCC refinement iteration")
    ax.set_ylabel("Mean accepted frozen local objective")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lambda_objective_history.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    save_json(results, out_dir / "lambda_ablation.json")

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
