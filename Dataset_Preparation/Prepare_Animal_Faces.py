"""
Downloads the Animal Faces dataset. I downsized this one in the model training file and the experiment files, so we don't downsize here. You could downsize here and not downsize later if you wanted.
"""

import os

try:
    import kagglehub
except ImportError:
    raise ImportError(
        "kagglehub not installed. Install with: pip install kagglehub"
    )

# Kaggle dataset slug
DATASET_SLUG = "andrewmvd/animal-faces"

def download_animal_faces():
    print("Downloading Animal Faces dataset from Kaggle...")

    dataset_path = kagglehub.dataset_download(DATASET_SLUG)

    print("\nDownload complete!")
    print(f"Dataset stored at:\n{dataset_path}")

    return dataset_path

if __name__ == "__main__":
    path = download_animal_faces()

    # Optional: inspect folder structure
    print("\nContents:")
    print(os.listdir(path))
