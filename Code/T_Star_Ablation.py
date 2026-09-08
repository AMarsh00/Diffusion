#!/usr/bin/env python3
"""
HCC_TStar_Ablation.py

t* sensitivity ablation for the CURRENT production High_Confidence_Curves.py.

IMPORTANT
---------
This script does NOT patch or replace the localization code.

It uses High_Confidence_Curves.py exactly as implemented, including the
finite-sample importance-sampling estimator

    Z_ij ~ N(z_i, sigma^2 I)
    w_ij ∝ exp(-||Z_ij||^2 / 2)

which targets the manuscript's localized measure while preserving the
finite-N behavior of the earlier successful HCC experiments.

The only quantity changed between conditions is proxy_t = t*.

Default sweep:
    t* in {200, 300, 400, 500, 600}

Everything else matches the current HCC demo:
    metric_lambda = 1e6
    exp ODE steps = 6
    local shooting iters = 3
    local shooting lr = 0.5
    curve points = 10
    inversion steps = 1000
    refinement DDIM steps = 100
    display DDIM steps = 1000
    sigma = 0.20
    candidates = 16
    HCC iterations = 50
    latent step = 0.18
    decay = 0.80
    minimum step = 0.035
    pair chunk = 16
    backtracking factor = 0.5
    max backtracking trials = 4

Outputs:
    tstar_ablation.png
    tstar_objective_history.png
    tstar_ablation.json

Put this file in the same Code/ directory as High_Confidence_Curves.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import High_Confidence_Curves as hcc


DEFAULT_DOG = (
    "/data5/accounts/marsh/.cache/kagglehub/datasets/andrewmvd/"
    "animal-faces/versions/1/afhq/train/dog/flickr_dog_000070.jpg"
)
DEFAULT_CAT = (
    "/data5/accounts/marsh/.cache/kagglehub/datasets/andrewmvd/"
    "animal-faces/versions/1/afhq/train/cat/flickr_cat_000070.jpg"
)

PAPER = {
    "metric_lambda": 1e6,
    "exp_ode_steps": 6,
    "local_shooting_iters": 3,
    "local_shooting_lr": 0.5,
    "n_geo_points": 10,
    "inversion_steps": 1000,
    "refine_decode_steps": 100,
    "display_decode_steps": 1000,
    "sigma": 0.20,
    "n_candidates": 16,
    "refinement_iters": 50,
    "latent_step_size": 0.18,
    "step_decay": 0.80,
    "min_latent_step": 0.035,
    "pair_chunk_size": 16,
    "backtracking_factor": 0.5,
    "max_backtracking_steps": 4,
    "energy_decrease_tol": 0.0,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dog-image", default=DEFAULT_DOG)
    p.add_argument("--cat-image", default=DEFAULT_CAT)
    p.add_argument(
        "--checkpoint",
        default="./vp_diffusion_outputs/unet_animal_epoch_2000.pt",
    )
    p.add_argument("--output-dir", default="./outputs/tstar_ablation")
    p.add_argument(
        "--tstar-values",
        nargs="+",
        type=int,
        default=[200, 300, 400, 500, 600],
        help="Default: 200 300 400 500 600.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def decode_display(model, scheduler, z_T):
    T = scheduler.num_timesteps - 1
    return hcc.ddim_reverse_segment(
        model,
        scheduler,
        z_T,
        T,
        0,
        num_steps=PAPER["display_decode_steps"],
    )


def run_hcc_condition(model, scheduler, z_pair_T, proxy_t: int, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    T = scheduler.num_timesteps - 1

    @torch.no_grad()
    def latent_to_clean_fn(z_T):
        return hcc.ddim_reverse_segment(
            model,
            scheduler,
            z_T,
            T,
            0,
            num_steps=PAPER["refine_decode_steps"],
        )

    def latent_to_clean_grad_fn(z_T):
        return hcc.ddim_reverse_segment_differentiable(
            model,
            scheduler,
            z_T,
            T,
            0,
            num_steps=PAPER["refine_decode_steps"],
            use_checkpoint=True,
        )

    def exp_map_fn(x, v):
        return hcc.geodesic_exp_map_proxy(
            x,
            v,
            model=model,
            scheduler=scheduler,
            proxy_t=int(proxy_t),
            metric_lambda=PAPER["metric_lambda"],
            n_steps=PAPER["exp_ode_steps"],
        )

    curve_z0 = hcc.build_linear_latent_curve(
        z_pair_T[0:1],
        z_pair_T[1:2],
        n_steps=PAPER["n_geo_points"],
    )

    (
        history_z,
        energy_history,
        accepted_energy_history,
        step_history,
    ) = hcc.refine_latent_constrained_clean_frechet(
        curve_z0,
        latent_to_clean_fn=latent_to_clean_fn,
        latent_to_clean_grad_fn=latent_to_clean_grad_fn,
        exp_map_fn=exp_map_fn,
        model=model,
        scheduler=scheduler,
        proxy_t=int(proxy_t),
        metric_lambda=PAPER["metric_lambda"],
        sigma=PAPER["sigma"],
        n_iters=PAPER["refinement_iters"],
        n_candidates=PAPER["n_candidates"],
        latent_step_size=PAPER["latent_step_size"],
        step_decay=PAPER["step_decay"],
        min_latent_step=PAPER["min_latent_step"],
        local_shooting_iters=PAPER["local_shooting_iters"],
        local_shooting_lr=PAPER["local_shooting_lr"],
        pair_chunk_size=PAPER["pair_chunk_size"],
        normalize_step=True,
        backtracking_factor=PAPER["backtracking_factor"],
        max_backtracking_steps=PAPER["max_backtracking_steps"],
        energy_decrease_tol=PAPER["energy_decrease_tol"],
    )

    return {
        "initial_z": curve_z0.detach(),
        "final_z": history_z[-1].detach(),
        "history_z": history_z,
        "energy_history": energy_history,
        "accepted_energy_history": accepted_energy_history,
        "step_history": step_history,
    }


@torch.no_grad()
def save_ablation_grid(lerp_x, condition_rows, output_path: Path):
    rows = [("Latent LERP", lerp_x)] + condition_rows
    nrows = len(rows)
    ncols = lerp_x.shape[0]

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(1.55 * ncols, 1.65 * nrows),
    )
    if nrows == 1:
        axes = axes[None, :]

    for r, (label, curve_x) in enumerate(rows):
        imgs = (curve_x.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
        for c in range(ncols):
            axes[r, c].imshow(transforms.ToPILImage()(imgs[c]))
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(label, rotation=0, labelpad=55, va="center")

    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 80)
    print("HCC t* ablation")
    print("=" * 80)
    print(f"device={device}")
    print(f"t* values={args.tstar_values}")
    print(f"lambda={PAPER['metric_lambda']:g}")
    print("localization estimator=current High_Confidence_Curves.py")
    print("proposal N(z_i, sigma^2 I) + self-normalized importance weights")
    print(
        f"sigma={PAPER['sigma']}, candidates={PAPER['n_candidates']}, "
        f"refinement_iters={PAPER['refinement_iters']}"
    )

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    model = hcc.UNetSD().to(device)
    scheduler = hcc.VPScheduler(num_timesteps=1000)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    x_dog = hcc.load_image(args.dog_image).to(device)
    x_cat = hcc.load_image(args.cat_image).to(device)
    x_pair = torch.cat([x_dog, x_cat], dim=0)

    T = scheduler.num_timesteps - 1

    print("\nInverting endpoints ONCE...")
    z_pair_T = hcc.ddim_forward_inversion_segment(
        model,
        scheduler,
        x_pair,
        0,
        T,
        num_steps=PAPER["inversion_steps"],
    )

    lerp_z = hcc.build_linear_latent_curve(
        z_pair_T[0:1],
        z_pair_T[1:2],
        n_steps=PAPER["n_geo_points"],
    )
    lerp_x = decode_display(model, scheduler, lerp_z)

    results = {
        "dog_image": args.dog_image,
        "cat_image": args.cat_image,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "parameters_except_tstar": PAPER,
        "localization_estimator": (
            "Current High_Confidence_Curves.py: "
            "Z~N(z_i,sigma^2 I), self-normalized importance weights "
            "w proportional exp(-||Z||^2/2)"
        ),
        "conditions": {},
    }

    condition_rows = []
    accepted_histories = {}

    for tstar in args.tstar_values:
        print("\n" + "=" * 80)
        print(f"RUNNING t*={tstar}")
        print("=" * 80)

        run = run_hcc_condition(
            model=model,
            scheduler=scheduler,
            z_pair_T=z_pair_T,
            proxy_t=tstar,
            seed=args.seed,
        )

        final_x = decode_display(model, scheduler, run["final_z"])
        condition_rows.append((rf"$t^*={tstar}$", final_x))

        pre_hist = [
            float(x.double().mean().item())
            for x in run["energy_history"]
        ]
        accepted_hist = [
            float(x.double().mean().item())
            for x in run["accepted_energy_history"]
        ]

        accepted_histories[str(tstar)] = accepted_hist
        results["conditions"][str(tstar)] = {
            "mean_pre_objective_history": pre_hist,
            "mean_accepted_objective_history": accepted_hist,
            "final_node_steps": (
                run["step_history"][-1].tolist()
                if run["step_history"]
                else []
            ),
        }

    save_ablation_grid(
        lerp_x,
        condition_rows,
        out_dir / "tstar_ablation.png",
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for tstar, hist in accepted_histories.items():
        if hist:
            ax.plot(
                range(1, len(hist) + 1),
                hist,
                label=rf"$t^*={int(tstar)}$",
            )

    ax.set_xlabel("HCC refinement iteration")
    ax.set_ylabel("Mean accepted frozen local objective")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        out_dir / "tstar_objective_history.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    (out_dir / "tstar_ablation.json").write_text(
        json.dumps(results, indent=2)
    )

    print("\nSaved:")
    print(" ", out_dir / "tstar_ablation.png")
    print(" ", out_dir / "tstar_objective_history.png")
    print(" ", out_dir / "tstar_ablation.json")


if __name__ == "__main__":
    main()
