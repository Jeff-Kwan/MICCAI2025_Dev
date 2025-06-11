import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader


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

def process_dataset(images_dir, labels_dir, out_image_dir, out_label_dir, pixdim):
    # create output dirs
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    # define the validation transform
    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS", lazy=True),
            mt.Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bicubic", "nearest"),
                lazy=True,
            ),
            mt.EnsureTyped(
                keys=["image", "label"],
                dtype=[torch.float32, torch.uint8],
                track_meta=True,
            ),
            mt.ThresholdIntensityd(
                keys=["label"],
                above=False,
                threshold=14,   # 14 classes
                cval=0,
            ),
            mt.ThresholdIntensityd( # upper bound 99.5%
                keys=["image"],
                above=False,
                threshold=295.0,
                cval=295.0,
            ),
            mt.ThresholdIntensityd( # lower bound 0.5%
                keys=["image"],
                above=True,
                threshold=-974.0, 
                cval=-974.0,
            ),
            mt.NormalizeIntensityd( # z-score normalization
                keys=["image"],
                subtrahend=77.515,
                divisor=142.119,
            ),
        ]
    )

    # build the MONAI dataset
    dataset = Dataset(data=get_data_files(images_dir, labels_dir), transform=transform)
    dataloader = ThreadDataLoader(
        dataset,
        batch_size=1,
        num_workers=32,
    )
    save = mt.Compose([
            mt.SaveImaged(
                keys=["image"],
                output_dir=out_image_dir,
                output_postfix="",
                output_ext=".nii.gz",
                separate_folder=False,
                output_dtype=torch.float32,
                print_log=False),
            mt.SaveImaged(
                keys=["label"],
                output_dir=out_label_dir,
                output_postfix="",
                output_ext=".nii.gz",
                separate_folder=False,
                output_dtype=torch.uint8,
                print_log=False),
    ])


    # iterate, transform, and save
    for batch in tqdm(dataloader, desc="Processing images"):
        label = batch["label"].numpy()
        if np.all(label == 0):
            continue  # Do not save empty labels data pair
        else:
            save(batch)


def process_labels(images_dir, labels_dir, out_label_dir, pixdim):
    # create output dirs
    os.makedirs(out_label_dir, exist_ok=True)

    # define the validation transform
    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["label"], ensure_channel_first=True),
            mt.Orientationd(keys=["label"], axcodes="RAS", lazy=True),
            mt.Spacingd(
                keys=["label"],
                pixdim=pixdim,
                mode=("nearest"),
                lazy=True,
            ),
            mt.EnsureTyped(
                keys=["label"],
                dtype=[torch.uint8],
                track_meta=True,
            ),
            mt.ThresholdIntensityd(
                keys=["label"],
                above=False,
                threshold=14,   # 14 classes
                cval=0,
            ),
        ]
    )

    # build the MONAI dataset
    dataset = Dataset(data=get_data_files(images_dir, labels_dir), transform=transform)
    dataloader = ThreadDataLoader(
        dataset,
        batch_size=1,
        num_workers=32,
    )
    save = mt.SaveImaged(
                keys=["label"],
                output_dir=out_label_dir,
                output_postfix="",
                output_ext=".nii.gz",
                separate_folder=False,
                output_dtype=torch.uint8,
                print_log=False)


    # iterate, transform, and save
    for batch in tqdm(dataloader, desc="Processing images"):
        label = batch["label"].numpy()
        if np.all(label == 0):
            continue
        else:
            save(batch)


if __name__ == "__main__":
    pixdim = (0.8, 0.8, 1.0)
    dir_list = [
        (
            "data/FLARE-Task2-LaptopSeg/train_gt_label/imagesTr",
            "data/FLARE-Task2-LaptopSeg/train_gt_label/labelsTr",
            "data/preprocessed/train_gt/images",
            "data/preprocessed/train_gt/labels",
        ),
        (
            "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Images",
            "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Labels",
            "data/preprocessed/val/images",
            "data/preprocessed/val/labels",
        ),
        (
            "data/FLARE-Task2-LaptopSeg/train_pseudo_label/imagesTr",
            "data/FLARE-Task2-LaptopSeg/train_pseudo_label/flare22_aladdin5_pseudo",
            "data/preprocessed/train_pseudo/images",
            "data/preprocessed/train_pseudo/aladdin5",
        ),
    ]

    for dirs in dir_list:
        process_dataset(*dirs, pixdim)

    process_labels(
        "data/FLARE-Task2-LaptopSeg/train_pseudo_label/imagesTr",
        "data/FLARE-Task2-LaptopSeg/train_pseudo_label/pseudo_label_blackbean_flare22",
        "data/preprocessed/train_pseudo/blackbean",
        pixdim)
