#!/usr/bin/env python3
"""Run the paper High-Confidence Curve experiment for one endpoint pair.

Example
-------
python Code/Run_HCC.py \
  --image-a /path/to/dog.jpg \
  --image-b /path/to/cat.jpg \
  --checkpoint "Model/Model Weights/unet_animal_epoch_2000.pt" \
  --config configs/afhq_paper.yaml \
  --output-dir outputs/hcc_pair
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from experiment_utils import (
    choose_device,
    decode_display,
    final_importance_ess,
    invert_pair,
    load_config,
    mean_history,
    run_hcc_from_latents,
    save_curve_grid,
    save_json,
    seed_everything,
    strict_load_model,
    summarize_step_history,
)
import High_Confidence_Curves as hcc


def parse_args():
    p = argparse.ArgumentParser(description="Run one High-Confidence Curve experiment")
    p.add_argument("--image-a", required=True)
    p.add_argument("--image-b", required=True)
    p.add_argument("--config", default="configs/afhq_paper.yaml")
    p.add_argument("--checkpoint", default=None,
                   help="Overrides checkpoint from config")
    p.add_argument("--output-dir", default="outputs/hcc_pair")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--row-iters", nargs="+", type=int, default=[0, 1, 5, 10, 25, 50],
        help="Refinement iterations to render as rows (values above the run length are skipped).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    seed_everything(seed)
    device = choose_device(args.device)
    checkpoint = args.checkpoint or cfg.get("checkpoint")
    if checkpoint is None:
        raise ValueError("No checkpoint provided in --checkpoint or configuration")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, scheduler = strict_load_model(
        checkpoint, device, num_timesteps=int(cfg.get("num_timesteps", 1000))
    )
    image_size = int(cfg.get("image_size", 64))
    x_a = hcc.load_image(args.image_a, image_size=image_size).to(device)
    x_b = hcc.load_image(args.image_b, image_size=image_size).to(device)
    x_pair = torch.cat([x_a, x_b], dim=0)

    t0 = time.perf_counter()
    print("Inverting endpoints...")
    z_pair = invert_pair(model, scheduler, x_pair, cfg)
    initial_z = hcc.build_linear_latent_curve(
        z_pair[0:1], z_pair[1:2], n_steps=int(cfg["n_geo_points"])
    )
    initial_x = decode_display(model, scheduler, initial_z, cfg)

    print("Running HCC refinement...")
    run = run_hcc_from_latents(model, scheduler, z_pair, cfg, seed=seed)
    final_x = decode_display(model, scheduler, run["final_z"], cfg)
    runtime = time.perf_counter() - t0

    save_curve_grid(
        [("Latent LERP", initial_x), ("HCC final", final_x)],
        out_dir / "hcc_initial_vs_final.png",
    )

    row_indices = sorted(set(i for i in args.row_iters if 0 <= i < len(run["history_z"])))
    rows = []
    for i in row_indices:
        curve_x = decode_display(model, scheduler, run["history_z"][i], cfg)
        rows.append((f"HCC iter {i}", curve_x))
    if rows:
        save_curve_grid(rows, out_dir / "hcc_refinement_rows.png")

    pre_hist = mean_history(run["energy_history"])
    accepted_hist = mean_history(run["accepted_energy_history"])
    step_summary = summarize_step_history(run["step_history"])
    ess = final_importance_ess(run["final_z"], cfg, seed + 100003)

    results = {
        "image_a": str(Path(args.image_a).resolve()),
        "image_b": str(Path(args.image_b).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "seed": seed,
        "device": str(device),
        "runtime_seconds": runtime,
        "parameters": cfg,
        "mean_pre_objective_history": pre_hist,
        "mean_accepted_objective_history": accepted_hist,
        "step_summary": step_summary,
        "final_importance_ess": ess,
        "surrogate_geometry_note": (
            "Image-scale logarithms use the symmetry-reduced Eq. (8) surrogate; "
            "accepted updates are checked by frozen-objective backtracking."
        ),
    }
    save_json(results, out_dir / "metrics.json")

    print(f"Saved {out_dir / 'hcc_initial_vs_final.png'}")
    if rows:
        print(f"Saved {out_dir / 'hcc_refinement_rows.png'}")
    print(f"Saved {out_dir / 'metrics.json'}")
    print(f"Runtime: {runtime:.1f}s")
    print(f"Acceptance fraction: {step_summary['acceptance_fraction']}")
    print(f"Final mean ESS: {ess['mean']}")


if __name__ == "__main__":
    main()
