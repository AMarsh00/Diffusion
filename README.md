# Confidence Metric for Diffusion Models

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.1-red.svg)
![License](https://img.shields.io/badge/license-AGPL-green.svg)

This repository implements the **confidence metric for diffusion models**, as described in our paper. The code enables computing **high-confidence latent curves**, **geodesic paths**, and reproducing experiments on both real images (CelebA-HQ) and synthetic Gaussian data.  

---

## 📂 Repository Structure

- **Data preparation and training**
  - `Prepare_Dataset.py`: Prepare the 64x64 CelebA-HQ dataset for training.
  - `Train_Model.py`: Train the UNet diffusion model.

- **Pre-trained model**
  - `unet_epoch_2000.pt`: Pre-trained weights for the 64x64 CelebA-HQ model. Can be downloaded to avoid retraining.

- **Example experiments (paper figures)**
  - `Graphic.py`, `Graphic2.py`, `Graphic3.py`: Scripts for reproducing Gaussian-based experiments from the paper.

- **Confidence metric & high-confidence curves**
  - `Confidence.py`: Core implementation of the confidence metric.
  - `High_Confidence_Curves.py`: Generates high-confidence latent curves.
  - `ConfChains.py`: Computes chains of high-confidence points for latent traversals.

---

## ⚙️ Installation & Dependencies

The code was tested on **Python 3.13**. Install dependencies with:

```bash
pip install torch torchvision matplotlib numpy scipy pillow tqdm
