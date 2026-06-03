"""
Script to download and prepare (downsize) CelebA-HQ for processing in our model. We downsize so that the model can be trained faster, downsizing is not necessary.
"""

import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Optional: auto-download via kagglehub
try:
    import kagglehub
except ImportError:
    kagglehub = None

# =========================
# Config
# =========================
DATASET_SLUG = "badasstechie/celebahq-resized-256x256"
OUTPUT_DIR = "./celeba_hq_prepared" # Replace with your directory if desired
TARGET_RESOLUTION = 64

# =========================
# Download dataset if needed
# =========================
def get_dataset_path():
    """
    Downloads CelebA-HQ via kagglehub if not already cached,
    and returns the local dataset directory.
    """
    if kagglehub is None:
        raise ImportError(
            "kagglehub is not installed. Install with: pip install kagglehub"
        )

    print("Checking / downloading dataset via kagglehub...")

    dataset_path = kagglehub.dataset_download(DATASET_SLUG)

    print(f"Dataset available at: {dataset_path}")
    return dataset_path

# =========================
# Image transform
# =========================
transform = transforms.Compose([
    transforms.Resize((TARGET_RESOLUTION, TARGET_RESOLUTION)),
])

# =========================
# Preprocessing
# =========================
def prepare_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"Found {len(image_files)} images to process.")

    for i, fname in enumerate(tqdm(image_files)):
        path = os.path.join(input_dir, fname)

        try:
            img = Image.open(path).convert("RGB")
            img = transform(img)

            save_path = os.path.join(output_dir, f"{i:06d}.png")
            img.save(save_path)

        except Exception as e:
            print(f"Skipping {fname}: {e}")

    print(f"Saved processed images to: {output_dir}")

if __name__ == "__main__":
    input_dir = get_dataset_path()
    prepare_dataset(input_dir, OUTPUT_DIR)
