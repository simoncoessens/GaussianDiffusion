"""
download_celeba.py -- Download CelebA dataset for encoding.

Downloads the aligned & cropped CelebA images using torchvision.
If torchvision download fails (Google Drive quota), provides manual instructions.

Output: data/celeba64/raw/celeba/  (torchvision standard layout)

Usage:
    python scripts/celeba64/download.py
    python scripts/celeba64/download.py --data_dir data/celeba64/raw
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Download CelebA dataset.")
    parser.add_argument("--data_dir", type=str, default="data/celeba64/raw",
                        help="Root directory for torchvision CelebA download")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir

    # Check if already downloaded
    img_dir = data_dir / "celeba" / "img_align_celeba"
    if img_dir.exists():
        import os
        n_files = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        if n_files >= 202599:
            print(f"CelebA already downloaded: {n_files} images in {img_dir}")
            return
        elif n_files > 0:
            print(f"Partial download found: {n_files}/202599 images in {img_dir}")

    print(f"Downloading CelebA to {data_dir} ...")
    print("This downloads ~1.3 GB from Google Drive.\n")

    try:
        from torchvision.datasets import CelebA

        # This downloads img_align_celeba.zip, identity, attributes, partition files
        ds = CelebA(root=str(data_dir), split="all", download=True)
        print(f"\nDownload complete! {len(ds)} images available.")
        print(f"Location: {data_dir / 'celeba'}")

    except Exception as e:
        print(f"\n[ERROR] torchvision download failed: {e}\n")
        print("=" * 60)
        print("  Manual download instructions:")
        print("=" * 60)
        print(f"""
CelebA is hosted on Google Drive, which has download quotas.
You can download it manually:

Option 1 — Kaggle (recommended):
  pip install kaggle
  kaggle datasets download -d jessicali9530/celeba-dataset
  unzip celeba-dataset.zip -d {data_dir}/celeba/

Option 2 — Direct download:
  1. Go to https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg
  2. Download img_align_celeba.zip
  3. Download list_eval_partition.txt
  4. Extract to: {data_dir}/celeba/img_align_celeba/

Option 3 — Academic torrent:
  Search for "CelebA" on https://academictorrents.com/

After downloading, the directory structure should be:
  {data_dir}/celeba/
    img_align_celeba/
      000001.jpg
      000002.jpg
      ...
      202599.jpg
    list_eval_partition.txt
""")
        sys.exit(1)


if __name__ == "__main__":
    main()
