import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader
import matplotlib.pyplot as plt

def get_data_files(labels_dir, extension = ".nii.gz"):
    """
    Returns a list of dicts with file paths for images and labels.
    Each dict has the keys "image" and "label".

    Raises:
        FileNotFoundError: if either directory does not exist.
        RuntimeError: if no files with the given extension are found.
        ValueError: if any image is missing a matching label.
    """
    labels_dir = Path(labels_dir)

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels_dir!r}")

    # Scan label directory once, build a set of names
    label_names = {
        entry.name
        for entry in os.scandir(labels_dir)
        if entry.is_file() and entry.name.endswith(extension)
    }
    if not label_names:
        raise RuntimeError(f"No '{extension}' files found in {labels_dir!r}")

    # Build result list
    return [
        {"label": str(labels_dir / name), 
         "base_name": name.removesuffix(".nii.gz")}
        for name in label_names
    ]

def process_labels(datafiles, num_classes=14):
    """
    Process the label images, compute histogram of label frequencies,
    and plot the histogram.
    """
    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["label"], ensure_channel_first=True),
            mt.AsDiscreted(keys=["label"], argmax=True),
            mt.EnsureTyped(keys=["label"], dtype=torch.uint8),
        ]
    )

    dataset = Dataset(data=datafiles, transform=transform)
    dataloader = ThreadDataLoader(dataset, batch_size=1, num_workers=190)

    freq_list = []

    for data in tqdm(dataloader, desc="Processing labels"):
        label = data["label"][0].numpy()  # Shape: (C=1, H, W, D) or similar
        label = label.squeeze()  # Remove channel dim if exists

        # Count frequency of each label value in this label image
        counts = np.bincount(label.ravel(), minlength=14)
        freq = counts / counts.sum()
        freq_list.append(freq)

        # Print the name if it only contains label 0
        if np.all(label == 0):
            print("All labels are zero for image:", data['label'][0].meta['filename_or_obj'])

    return np.mean(np.array(freq_list), axis=0)

if __name__ == "__main__":
    datafiles = get_data_files(
        "data/small/train_pseudo/softquant")

    label_frequencies = torch.tensor(process_labels(datafiles)).squeeze()

    label_frequencies *= 100
    print("Label frequencies (%):", [f"{f:.4f}" for f in label_frequencies.tolist()])
