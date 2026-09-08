#!/usr/bin/env python3
"""
HCC_Lambda0_Ablation.py

Lambda=0 ablation for the CURRENT production High_Confidence_Curves.py.

IMPORTANT
---------
This script does NOT patch or replace the localization code.

It uses High_Confidence_Curves.py exactly as implemented, including the
finite-sample importance-sampling estimator

    Z_ij ~ N(z_i, sigma^2 I)
    w_ij ∝ exp(-||Z_ij||^2 / 2)

which targets the manuscript's localized measure while preserving the
finite-N behavior of the earlier successful HCC experiments.

The only quantity changed between conditions is metric_lambda.

Default comparison:
    lambda = 0
    lambda = 1e6

Everything else matches the current HCC demo:
    proxy_t = 400
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
    lambda0_ablation.png
    lambda0_objective_history.png
    lambda0_ablation.json

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


# ---------------------------------------------------------------------
# Defaults matching the current production HCC demo
# ---------------------------------------------------------------------
DEFAULT_DOG = (
    "/data5/accounts/marsh/.cache/kagglehub/datasets/andrewmvd/"
    "animal-faces/versions/1/afhq/train/dog/flickr_dog_000100.jpg"
)
DEFAULT_CAT = (
    "/data5/accounts/marsh/.cache/kagglehub/datasets/andrewmvd/"
    "animal-faces/versions/1/afhq/train/cat/flickr_cat_000100.jpg"
)

PAPER = {
    "proxy_t": 400,
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
    p.add_argument(
        "--output-dir",
        default="./outputs/lambda0_ablation",
    )

    p.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=[0.0, 1e6],
        help="Default: lambda=0 vs lambda=1e6.",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default=None,
        help="Default: cuda if available, else cpu.",
    )

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


def run_hcc_condition(
    model,
    scheduler,
    z_pair_T,
    metric_lambda: float,
    seed: int,
):
    """
    Run ONE HCC condition.

    Resetting the RNG here ensures that lambda=0 and lambda=1e6 receive
    the same proposal-noise sequence, so lambda is the only intended
    experimental difference.
    """
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
            proxy_t=PAPER["proxy_t"],
            metric_lambda=float(metric_lambda),
            n_steps=PAPER["exp_ode_steps"],
        )

    # Same initialization for every lambda condition.
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
        proxy_t=PAPER["proxy_t"],
        metric_lambda=float(metric_lambda),
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
            axes[r, c].imshow(
                transforms.ToPILImage()(imgs[c])
            )
            axes[r, c].axis("off")

        axes[r, 0].set_ylabel(
            label,
            rotation=0,
            labelpad=55,
            va="center",
        )

    plt.tight_layout()
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed seed before model/data setup.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 80)
    print("HCC lambda ablation")
    print("=" * 80)
    print(f"device={device}")
    print(f"lambda values={args.lambda_values}")
    print("localization estimator=current High_Confidence_Curves.py")
    print(
        "proposal N(z_i, sigma^2 I) + self-normalized importance weights"
    )
    print(
        f"proxy_t={PAPER['proxy_t']}, sigma={PAPER['sigma']}, "
        f"candidates={PAPER['n_candidates']}, "
        f"refinement_iters={PAPER['refinement_iters']}"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    model = hcc.UNetSD().to(device)
    scheduler = hcc.VPScheduler(num_timesteps=1000)

    state = torch.load(
        args.checkpoint,
        map_location=device,
    )
    model.load_state_dict(state)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Shared endpoints and inversion
    # ------------------------------------------------------------------
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

    # Baseline LERP uses the SAME inverted endpoints.
    lerp_z = hcc.build_linear_latent_curve(
        z_pair_T[0:1],
        z_pair_T[1:2],
        n_steps=PAPER["n_geo_points"],
    )
    lerp_x = decode_display(
        model,
        scheduler,
        lerp_z,
    )

    # ------------------------------------------------------------------
    # Run each lambda
    # ------------------------------------------------------------------
    results = {
        "dog_image": args.dog_image,
        "cat_image": args.cat_image,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "parameters": PAPER,
        "localization_estimator": (
            "Current High_Confidence_Curves.py: "
            "Z~N(z_i,sigma^2 I), self-normalized importance weights "
            "w proportional exp(-||Z||^2/2)"
        ),
        "conditions": {},
    }

    condition_rows = []
    accepted_histories = {}

    for lam in args.lambda_values:
        print("\n" + "=" * 80)
        print(f"RUNNING lambda={lam:g}")
        print("=" * 80)

        run = run_hcc_condition(
            model=model,
            scheduler=scheduler,
            z_pair_T=z_pair_T,
            metric_lambda=lam,
            seed=args.seed,
        )

        final_x = decode_display(
            model,
            scheduler,
            run["final_z"],
        )

        condition_rows.append(
            (rf"$\lambda={lam:g}$", final_x)
        )

        pre_hist = [
            float(x.double().mean().item())
            for x in run["energy_history"]
        ]
        accepted_hist = [
            float(x.double().mean().item())
            for x in run["accepted_energy_history"]
        ]

        accepted_histories[str(lam)] = accepted_hist

        results["conditions"][str(lam)] = {
            "mean_pre_objective_history": pre_hist,
            "mean_accepted_objective_history": accepted_hist,
            "final_node_steps": (
                run["step_history"][-1].tolist()
                if run["step_history"]
                else []
            ),
        }

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    save_ablation_grid(
        lerp_x,
        condition_rows,
        out_dir / "lambda0_ablation.png",
    )

    # Objective histories are not directly comparable in scale between
    # different lambda values, but plotting each trajectory is still useful
    # for showing within-condition numerical behavior.
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for lam, hist in accepted_histories.items():
        if hist:
            ax.plot(
                range(1, len(hist) + 1),
                hist,
                label=rf"$\lambda={float(lam):g}$",
            )

    ax.set_xlabel("HCC refinement iteration")
    ax.set_ylabel("Mean accepted frozen local objective")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        out_dir / "lambda0_objective_history.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Save machine-readable results
    # ------------------------------------------------------------------
    (out_dir / "lambda0_ablation.json").write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    print("\nSaved:")
    print(" ", out_dir / "lambda0_ablation.png")
    print(" ", out_dir / "lambda0_objective_history.png")
    print(" ", out_dir / "lambda0_ablation.json")


if __name__ == "__main__":
    main()
