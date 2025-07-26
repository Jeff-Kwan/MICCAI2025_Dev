import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader
import matplotlib.pyplot as plt

def get_data_files(labels_dir, extension=".nii.gz"):
    """
    Returns a list of dicts with file paths for labels only.
    Each dict has the key "label".

    Raises:
        FileNotFoundError: if directory does not exist.
        RuntimeError: if no files with the given extension are found.
    """
    labels_dir = Path(labels_dir)

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels_dir!r}")

    label_names = sorted(
        entry.name
        for entry in os.scandir(labels_dir)
        if entry.is_file() and entry.name.endswith(extension)
    )
    if not label_names:
        raise RuntimeError(f"No '{extension}' files found in {labels_dir!r}")

    return [
        {"label": str(labels_dir / name), "base_name": name.removesuffix(".nii.gz")}
        for name in label_names
    ]

def process_labels(datafiles, num_classes=14):
    """
    For each label class (excluding background), compute min, max, mean, median
    number of voxels across all images.
    """
    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["label"], ensure_channel_first=True),
            mt.Orientationd(keys=["label"], axcodes="RAS"),
            mt.Spacingd(keys=["label"], pixdim=(0.8, 0.8, 2.5), mode="nearest"),
            mt.EnsureTyped(keys=["label"], dtype=np.uint8, track_meta=True),
        ]
    )

    dataset = Dataset(data=datafiles, transform=transform)
    dataloader = ThreadDataLoader(dataset, batch_size=1, num_workers=6)

    # Store per-image voxel counts for each class (excluding background)
    per_class_voxel_counts = [[] for _ in range(num_classes)]

    for data in tqdm(dataloader, desc="Processing labels"):
        label = data["label"].squeeze().numpy()

        # Count voxels for each class in this image
        counts = np.bincount(label.ravel(), minlength=num_classes)
        for cls in range(num_classes):
            if counts[cls] > 0:
                per_class_voxel_counts[cls].append(counts[cls])

    # Compute stats for each class (excluding background)
    print("Min voxel count and digit length per label class (excluding background):")
    for cls in range(1, num_classes):  # skip background class 0
        arr = np.array(per_class_voxel_counts[cls])
        if arr.size == 0:
            print(f"Class {cls}: min=N/A, digits=N/A")
        else:
            min_val = int(arr.min())
            num_digits = len(str(min_val))
            print(f"Class {cls}: min={min_val}, digits-1={num_digits-1}")

    return

if __name__ == "__main__":
    datafiles = get_data_files(
        "archived_code/plots/labels/Validation-Public-Labels")

    process_labels(datafiles)
