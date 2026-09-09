# Refining Diffusion-Model Interpolations with Local Fréchet Statistics

Official PyTorch implementation of **High-Confidence Curves (HCC)**.

The method refines latent interpolation points using localized Fréchet-motivated directions under the score-induced Riemannian metric

\[
g(x)=I+\lambda \widetilde s_\theta(x)\widetilde s_\theta(x)^\top.
\]

For image-scale experiments, the exact Levi-Civita geometry is **not** claimed to be solved directly. The implementation uses the symmetry-reduced Eq. (8) flow as an explicitly defined computational surrogate for exponential/logarithm calculations. HCC then verifies accepted finite-sample updates using frozen-objective backtracking.

The image implementation uses the same localized target distribution as the manuscript, but estimates it with self-normalized importance sampling from a Gaussian proposal centered at the current latent point. This preserves the finite-sample implementation used for the reported image experiments.

## Repository layout

```text
.
├── Code/
│   ├── High_Confidence_Curves.py     # Core HCC implementation
│   ├── Run_HCC.py                    # One-pair paper experiment
│   ├── Geometry_Diagnostics.py       # Eq. (8) numerical validation
│   ├── Lambda_Ablation.py            # lambda = 0, 1e4, 1e5, 1e6 by default
│   ├── T_Star_Ablation.py            # t* = 200, ..., 600
│   ├── Geodesics.py                  # Eq. (8) surrogate shooting baseline
│   ├── NEB_Method.py                 # score-force NEB-style heuristic
│   ├── Confidence.py                 # local geometric-centrality demo
│   └── experiment_utils.py           # shared experiment helpers
├── configs/
│   └── afhq_paper.yaml               # exact paper-facing HCC configuration
├── tests/
│   └── test_hcc_core.py
├── Dataset_Preparation/
│   ├── Prepare_CelebA.py
│   └── Prepare_Animal_Faces.py
├── Model/
│   ├── Train_Model_CelebA.py
│   ├── Train_Model_Animal_Faces.py
│   └── Model Weights/
│       ├── unet_epoch_2000.pt
│       └── unet_animal_epoch_2000.pt
├── 2D_Examples/
├── environment.yml
├── requirements.txt
└── README.md
```

## Installation

Python 3.11 is recommended. Install a PyTorch/torchvision build appropriate for your machine first, then install the remaining dependencies.

```bash
conda create -n hcc python=3.11 -y
conda activate hcc

# Install torch/torchvision for your CPU or CUDA setup using the official
# PyTorch instructions, then:
pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate hcc
# Then install the appropriate torch/torchvision build for your machine.
```

The code **never silently falls back to random weights** when a checkpoint is missing.

## Paper configuration

All main AFHQ HCC settings are stored in:

```text
configs/afhq_paper.yaml
```

The configuration includes the proxy timestep, metric weight, surrogate ODE resolution, shooting budget, DDIM inversion/rendering budgets, localization scale, number of local candidates, refinement iterations, step schedule, and backtracking parameters.

The lightweight defaults inside `High_Confidence_Curves.py` are API defaults only; the paper-facing scripts read `configs/afhq_paper.yaml`.

## 1. Run one HCC interpolation

```bash
python Code/Run_HCC.py \
  --image-a /path/to/dog.jpg \
  --image-b /path/to/cat.jpg \
  --checkpoint "Model/Model Weights/unet_animal_epoch_2000.pt" \
  --config configs/afhq_paper.yaml \
  --output-dir outputs/hcc_pair
```

Outputs include:

- `hcc_initial_vs_final.png`
- `hcc_refinement_rows.png`
- `metrics.json`

`metrics.json` records the exact configuration, runtime, accepted frozen-objective history, update acceptance fraction, and final importance-sampling ESS.

## 2. Validate the Eq. (8) surrogate numerically

```bash
python Code/Geometry_Diagnostics.py \
  --checkpoint "Model/Model Weights/unet_animal_epoch_2000.pt" \
  --config configs/afhq_paper.yaml \
  --output-dir outputs/geometry_diagnostics
```

This reports:

- shooting endpoint RMSE and success fraction,
- sensitivity to 6 / 12 / 24 / 48 surrogate ODE steps,
- relative Jacobian asymmetry `||Jv - J^T v||`,
- omitted antisymmetric acceleration magnitude relative to the retained and full Eq. (7) accelerations.

The purpose is to characterize the surrogate numerically, not to claim exact equivalence with the full Levi-Civita dynamics.

## 3. Lambda ablation

```bash
python Code/Lambda_Ablation.py \
  --image-a /path/to/dog.jpg \
  --image-b /path/to/cat.jpg \
  --checkpoint "Model/Model Weights/unet_animal_epoch_2000.pt" \
  --lambda-values 0 1e4 1e5 1e6 \
  --output-dir outputs/lambda_ablation
```

All conditions use the same inverted endpoints and reset the RNG so the same proposal-noise sequence is used across lambda values.

## 4. Proxy-timestep ablation

```bash
python Code/T_Star_Ablation.py \
  --image-a /path/to/dog.jpg \
  --image-b /path/to/cat.jpg \
  --checkpoint "Model/Model Weights/unet_animal_epoch_2000.pt" \
  --tstar-values 200 300 400 500 600 \
  --output-dir outputs/tstar_ablation
```

## 5. Surrogate shooting baseline

```bash
python Code/Geodesics.py \
  --checkpoint "Model/Model Weights/unet_animal_epoch_2000.pt" \
  --lambda-values 0 1e4 1e5 1e6 \
  --output-dir outputs/surrogate_geodesics
```

This script generates two random diffusion-model endpoints and computes clean-coordinate paths using the symmetry-reduced Eq. (8) shooting flow. These are **surrogate shooting paths**, not claimed exact geodesics of Eq. (7) for a nonsymmetric learned score Jacobian.

## 6. Score-force NEB-style baseline

```bash
python Code/NEB_Method.py \
  --image-a /path/to/dog.jpg \
  --image-b /path/to/cat.jpg \
  --checkpoint "Model/Model Weights/unet_animal_epoch_2000.pt" \
  --n-iters 500 \
  --lr-values 1e-4 1e-3 1e-2 1e-1 \
  --k-values 0 1 \
  --output-dir outputs/neb_sweep
```

The learned score proxy is used directly as a force. Because the learned network is not assumed conservative, this is described as a **score-force NEB-style heuristic**, not canonical NEB on a known scalar energy landscape.

If a positive `--convergence-tol` is supplied, the run can terminate when the per-coordinate RMS chain update falls below that tolerance. Otherwise the script uses the requested fixed iteration budget.

## 7. Synthetic experiments

```bash
python 2D_Examples/Expected_Primitive_Updates.py
python 2D_Examples/High_Confidence_Curves.py
```

These reproduce the two-dimensional Gaussian experiments used for interpretable validation.

## Tests

```bash
pytest -q
```

The included tests check that:

- direct localized sampling matches the analytic Gaussian mean and variance,
- importance weights are normalized,
- HCC preserves endpoints,
- accepted frozen-objective updates do not increase the tested local objective,
- missing checkpoints fail loudly rather than running random weights.

## Reproducibility policy

For a paper result, save the generated JSON file together with the figure. The JSON contains the configuration and numerical diagnostics needed to identify the exact run. For a submission release, tag the repository commit used to generate the final manuscript figures.

## Terminology

- **Exact metric:** `g(x) = I + lambda s_tilde(x) s_tilde(x)^T`.
- **Exact theory:** Riemannian distance, logarithm, Fréchet mean/variance under the stated assumptions.
- **Image-scale numerical geometry:** symmetry-reduced Eq. (8) surrogate.
- **High-Confidence Curves:** generator-constrained interpolation refinement using localized Fréchet-motivated directions and frozen-objective backtracking.
- **Local geometric centrality:** the interpretation of the local mean/variance score. The code does not claim to globally maximize a single confidence functional.

## License

MIT.
