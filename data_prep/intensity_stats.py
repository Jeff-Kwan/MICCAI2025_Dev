import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader
import multiprocessing as mp

def get_data_files(images_dir, labels_dir, extension = ".nii.gz"):
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

def process_dataset(datafiles):
    # define the validation transform
    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.ThresholdIntensityd(
                keys=["label"],
                above=False,
                threshold=14,   # 14 classes
                cval=0),
            mt.EnsureTyped(
                keys=["image", "label"],
                dtype=[torch.float32, torch.uint8],
                track_meta=True),
        ]
    )

    # build the MONAI dataset
    dataset = Dataset(data=datafiles, transform=transform)
    dataloader = ThreadDataLoader(dataset, batch_size=1, num_workers=64,)

    foreground_intensities = []

    for data in tqdm(dataloader, desc="Processing images"):
        # Get foreground mask
        image = data["image"][0]
        label = data["label"][0]
        foreground_mask = label > 0

        # Extract foreground intensities
        foreground_intensities.append(image[foreground_mask].numpy().ravel())

    # Concatenate all foreground intensities into a single array
    all_foreground_intensities = np.concatenate(foreground_intensities)

    print(f"Total foreground intensities collected: {len(all_foreground_intensities)}")

    # Compute 0.5 and 99.5 percentiles
    p_low, p_high = np.percentile(all_foreground_intensities, [0.5, 99.5])

    # Clip the intensities
    clipped_intensities = np.clip(all_foreground_intensities, p_low, p_high)
    
    # Compute mean and std
    mean = np.mean(clipped_intensities)
    std = np.std(clipped_intensities)
    return p_low, p_high, mean, std



if __name__ == "__main__":
    datafiles = get_data_files("data/FLARE-Task2-LaptopSeg/train_gt_label/imagesTr", "data/FLARE-Task2-LaptopSeg/train_gt_label/labelsTr")
    datafiles += get_data_files("data/FLARE-Task2-LaptopSeg/train_pseudo_label/imagesTr", "data/FLARE-Task2-LaptopSeg/train_pseudo_label/flare22_aladdin5_pseudo")
    p_low, p_high, mean, std = process_dataset(datafiles)
    print(f"Final results: 0.5th percentile = {p_low}, 99.5th percentile = {p_high}, "
          f"Clipped mean = {mean}, Clipped std = {std}")

'''
Over 50 GT + 2000 Aladdin5 training set to get foreground intensities
Total foreground intensities collected: 5177351249
Final results: 0.5th percentile = -974.0, 99.5th percentile = 295.0, 
Clipped mean = 77.51521301269531, Clipped std = 142.11891174316406
'''