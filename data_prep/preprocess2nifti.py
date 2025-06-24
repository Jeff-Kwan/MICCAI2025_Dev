import os
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
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

def get_skipped_files(images_dir, labels_dir, skipped, extension = ".nii.gz"):
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
    names = [
        {"image": str(images_dir / name), "label": str(labels_dir / name), 
         "base_name": name.removesuffix(".nii.gz")}
        for name in image_names
    ]

    # Leave in only the skipped files
    skipped_files = [
        item for item in names if item["base_name"] in skipped
    ]
    assert len(skipped_files) == len(skipped), "Not all skipped files found in directories"
    return skipped_files


def process_dataset(file_getter, images_dir, labels_dir, out_image_dir, out_label_dir, pixdim):
    # create output dirs
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"),
            mt.SqueezeDimd(keys=["image", "label"], dim=0),
            mt.Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=(3, "nearest"),
                lazy=False,
            ),
            mt.EnsureChannelFirstd(keys=["image", "label"]),
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
            mt.ThresholdIntensityd( # upper bound 99.5% from Aladdin5 + GT stats
                keys=["image"],
                above=False,
                threshold=295.0,
                cval=295.0,
            ),
            mt.ThresholdIntensityd( # lower bound 0.5% from Aladdin5 + GT stats
                keys=["image"],
                above=True,
                threshold=-974.0, 
                cval=-974.0,
            ),
            mt.NormalizeIntensityd( # z-score normalization from GT stats
                keys=["image"],
                subtrahend=95.958,
                divisor=139.964,
            ),
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
                print_log=False)
        ]
    )

    # build the MONAI dataset
    dataset = Dataset(data=file_getter(images_dir, labels_dir), transform=transform)
    dataloader = ThreadDataLoader(
        dataset,
        batch_size=1,
        num_workers=128,
    )

    # iterate, transform, and save
    for batch in tqdm(dataloader, desc="Processing images"):
        pass

    return skipped



if __name__ == "__main__":
    pixdim = (0.8, 0.8, 2.5)
    dir_list = [
        (
            "data/FLARE-Task2-LaptopSeg/train_gt_label/imagesTr",
            "data/FLARE-Task2-LaptopSeg/train_gt_label/labelsTr",
            "data/nifti/train_gt/images",
            "data/nifti/train_gt/labels",
        ),
        (
            "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Images",
            "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Labels",
            "data/nifti/val/images",
            "data/nifti/val/labels",
        ),
        (
            "data/FLARE-Task2-LaptopSeg/train_pseudo_label/imagesTr",
            "data/FLARE-Task2-LaptopSeg/train_pseudo_label/flare22_aladdin5_pseudo",
            "data/nifti/train_pseudo/images",
            "data/nifti/train_pseudo/pseudo",
        ),
    ]

    skipped = []
    for dirs in dir_list:
        skipped += process_dataset(get_data_files, *dirs, pixdim)

    process_dataset(get_skipped_files,
                    (
            "data/FLARE-Task2-LaptopSeg/train_pseudo_label/imagesTr",
            "data/FLARE-Task2-LaptopSeg/train_pseudo_label/flare22_blackbean_pseudo",
            "data/nifti/train_pseudo/images",
            "data/nifti/train_pseudo/pseudo",
        ),)
    
