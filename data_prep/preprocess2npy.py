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

def process_dataset(images_dir, labels_dir, out_image_dir, out_label_dir, pixdim):
    # create output dirs
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    # define the validation transform
    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS", lazy=True),
            mt.SqueezeDimd(keys=["image", "label"],dim=0),
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
        num_workers=128,
    )

    crop = mt.CropForegroundd(
                keys=["image", "label"],
                source_key="label",
                margin=16, # Keep some margin
                allow_smaller=False),

    # iterate, transform, and save
    for batch in tqdm(dataloader, desc="Processing images"):
        if (batch["image"].shape[0] > 550 or 
            batch["image"].shape[1] > 550 or 
            batch["image"].shape[2] > 300):
            batch = crop(batch)
            print(f"Cropping {batch['base_name'][0]} to {batch['image'].shape}")

        img = batch["image"].numpy().squeeze().astype(np.float32)
        label = batch["label"].numpy().squeeze().astype(np.uint8)
        base_name = batch["base_name"][0]

        if label.any():
            # Save as npy
            np.save(os.path.join(out_image_dir, f"{base_name}.npy"), img)
            np.save(os.path.join(out_label_dir, f"{base_name}.npy"), label)
        else:
            print(f"Skipping {base_name} due to empty label")
            continue



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
