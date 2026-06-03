# A Geometrically-Based Confidence Metric for Diffusion Models

![Python](https://img.shields.io/badge/python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.12-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Official implementation of **"A Geometrically-Based Confidence Metric for Diffusion Models"**.

This repository introduces a confidence metric for diffusion models and a High-Confidence Curve interpolation based on it, which improves upon current interpolation algorithms.

Experiments and pretrained model weights are provided for both synthetic and image-based datasets, including CelebA-HQ and Animal Faces.

---

# Overview

Traditional interpolation methods often rely on Euclidean line segments between samples, which may traverse regions poorly supported by a diffusion model.

Our confidence metric instead quantifies how well a diffusion model supports a trajectory and enables the construction of paths that remain in regions of high model confidence.

---

# Repository Structure

```text
.
├── Code/
│   ├── Confidence.py
│   ├── Geodesics.py
│   ├── High_Confidence_Curves.py
│   └── NEB_Method.py
│
├── Dataset_Preparation/
│   ├── prepare_celeba.py
│   └── prepare_animals.py
│
├── 2D_Examples/
│   ├── Expected_Primitive_Updates.py
│   └── High_Confidence_Curves.py
│
├── Models/
|   └── Train_Model_Animal_Faces.py
|   └── Train_Model_CelebA.py
│   └── Model_Weights/
│       ├── unet_epoch_2000.pt
│       └── unet_animal_epoch_2000.pt
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Datasets

## CelebA-HQ

The repository includes scripts for downloading and preparing the CelebA-HQ dataset at 64×64 resolution.

## Animal Faces

Scripts are also provided for downloading and preparing the Animal Faces dataset.

Prepared datasets are used to reproduce the experiments reported in the paper.

---

# Pretrained Models

Pretrained diffusion model checkpoints are provided for reproducing experiments without retraining.

| Checkpoint                  | Dataset       | Resolution |
| --------------------------- | ------------- | ---------- |
| `unet_epoch_2000.pt`        | CelebA-HQ     | 64×64      |
| `unet_animal_epoch_2000.pt` | Animal Faces  | 64×64      |

Model checkpoints available in

```text
Models/Model_Weights/
```

---

# Installation

The code was tested using Python 3.14.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Quick Start

## Generate High-Confidence Curves

```bash
python Code/High_Confidence_Curves.py
```

Generates confidence-maximizing interpolation trajectories between training samples.

---

## Compute Diffusion Geodesics

```bash
python Code/Geodesics.py
```

Computes geodesic paths under the proposed confidence metric.

---

## Run NEB Optimization

```bash
python Code/NEB_Method.py
```

Uses the Nudged Elastic Band method to refine interpolation trajectories.

---

## Reproduce Synthetic Experiments

Expected primitive updates:

```bash
python 2D_Examples/Expected_Primitive_Updates.py
```

High-confidence curves:

```bash
python 2D_Examples/High_Confidence_Curves.py
```

These scripts reproduce the two-dimensional Gaussian experiments presented in the paper.

---

# Reproducing Results

1. Download and prepare the datasets.
2. Download pretrained checkpoints or train models from scratch.
3. Run the scripts in the `Code/` directory.
4. Run the synthetic experiments in `2D_Examples/`.
5. Compare generated trajectories and figures with those reported in the paper.

---

# Citation

If you use this repository in your research, please cite (no citation available yet, will add soon):

```bibtex
@article{}
```

---

# License

This repository is released under the MIT License.

---
