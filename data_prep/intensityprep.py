import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import monai.transforms as mt
from monai.data import Dataset, DataLoader
from tdigest import TDigest

def get_data_files(images_dir, extension = ".nii.gz"):
    """
    Returns a list of dicts with file paths for images and labels.
    Each dict has the keys "image" and "label".

    Raises:
        FileNotFoundError: if either directory does not exist.
        RuntimeError: if no files with the given extension are found.
        ValueError: if any image is missing a matching label.
    """
    images_dir = Path(images_dir)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir!r}")

    # Scan image directory
    image_names = sorted(
        entry.name
        for entry in os.scandir(images_dir)
        if entry.is_file() and entry.name.endswith(extension)
    )
    if not image_names:
        raise RuntimeError(f"No '{extension}' files found in {images_dir!r}")

    # Build result list
    return [
        {"image": str(images_dir / name)} for name in image_names
    ]


def get_thresholds():
    # Load transforms
    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image"]),
            mt.EnsureTyped(
                keys=["image"],
                dtype=[torch.float32],
                track_meta=False)
        ]
    )

    # build the MONAI dataset
    datafiles = get_data_files("data/preprocessed/train_gt/images") +\
                get_data_files("data/preprocessed/train_pseudo/images")
    dataset = Dataset(data=datafiles, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=64,
    )

    td = TDigest()

    for data in tqdm(dataloader, desc="Threshold"):
        td.update(data["image"].numpy().ravel())

    p_low  = td.percentile(0.5)   # 0.5th percentile
    p_high = td.percentile(99.5)  # 99.5th percentile
    return p_low, p_high

if __name__ == "__main__":
    p_low, p_high = get_thresholds()
    print(f"Low threshold: {p_low}, High threshold: {p_high}")
