import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader
import matplotlib.pyplot as plt

def get_data_files(images_dir, labels_dir, extension = ".npy"):
    """
    Returns a list of dicts with file paths for images and labels.
    Each dict has the keys "image" and "label".

    Raises:
        FileNotFoundError: if either directory does not exist.
        RuntimeError: if no files with the given extension are found.
        ValueError: if any image is missing a matching label.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir!r}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels_dir!r}")

    # Scan image directory
    image_names = sorted(
        entry.name
        for entry in os.scandir(images_dir)
        if entry.is_file() and entry.name.endswith(extension)
    )
    if not image_names:
        raise RuntimeError(f"No '{extension}' files found in {images_dir!r}")

    # Scan label directory once, build a set of names
    label_names = {
        entry.name
        for entry in os.scandir(labels_dir)
        if entry.is_file() and entry.name.endswith(extension)
    }
    if not label_names:
        raise RuntimeError(f"No '{extension}' files found in {labels_dir!r}")

    # Detect any missing labels in one go
    missing = [name for name in image_names if name not in label_names]
    if missing:
        missing_list = ", ".join(repr(n) for n in missing)
        raise ValueError(f"Missing labels for images: {missing_list}")

    # Build result list
    return [
        {"image": str(images_dir / name), "label": str(labels_dir / name), 
         "base_name": name.removesuffix(".nii.gz")}
        for name in image_names
    ]

def process_labels(datafiles, num_classes=14):
    """
    Process the label images, compute histogram of label frequencies,
    and plot the histogram.
    """
    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["label"]),
            mt.EnsureTyped(keys=["label"], dtype=torch.uint8, track_meta=True),
        ]
    )

    dataset = Dataset(data=datafiles, transform=transform)
    dataloader = ThreadDataLoader(dataset, batch_size=1, num_workers=128)

    label_counts = np.zeros(num_classes, dtype=np.int64)

    for data in tqdm(dataloader, desc="Processing labels"):
        label = data["label"][0].numpy()  # Shape: (C=1, H, W, D) or similar
        label = label.squeeze()  # Remove channel dim if exists

        # Count frequency of each label value in this label image
        counts = np.bincount(label.ravel(), minlength=num_classes)
        label_counts += counts

        # Print the name if it only contains label 0
        if np.all(label == 0):
            print("All labels are zero for image:", data['label'][0].meta['filename_or_obj'])

    # Plot the histogram
    # classes = np.arange(num_classes)
    # plt.bar(classes, label_counts)
    # plt.xlabel("Label Class")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Label Frequencies")
    # plt.xticks(classes)
    # plt.show()

    return label_counts

if __name__ == "__main__":
    datafiles = get_data_files(
        "data/preprocessed/train_gt/images",
        "data/preprocessed/train_gt/labels")
    # datafiles += get_data_files(
    #     "data/preprocessed/train_pseudo/images",
    #     "data/preprocessed/train_pseudo/aladdin5")
    label_counts = torch.tensor(process_labels(datafiles)).squeeze()
    # label_weights = label_counts.sum().item() / (label_counts * len(label_counts))
    # label_weights = torch.log(label_weights)
    # label_weights += label_weights.min().abs()+0.1
    # print("Label weights:", [f"{w:.4f}" for w in label_weights.tolist()])
    label_frequencies = label_counts / label_counts.sum()*100
    print("Label frequencies (%):", [f"{f:.4f}" for f in label_frequencies.tolist()])


'''
Analysis on Ground Truth Preprocessed data:

Label frequencies (%): [89.5178, 5.6570, 0.6980, 0.8370, 0.3288, 0.3280, 0.3000, 0.0154, 0.0184, 0.1188, 0.0555, 1.1621, 0.2652, 0.6981]

With background class weight set to 1, log-space difference as class weights
Label weights: [0.1000, 2.8616, 4.9539, 4.7724, 5.7069, 5.7091, 5.7985, 8.7701, 8.5915, 6.7250, 7.4852, 4.4442, 5.9218, 4.9539]

Then to 1 decimal place rounding:
[0.1, 2.9, 5.0, 4.8, 5.7, 5.7, 5.8, 8.8, 8.6, 6.7, 7.5, 4.4, 5.9, 5.0]
'''
