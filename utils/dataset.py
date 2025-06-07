import os
from pathlib import Path
from typing import List, Dict
import torch
import numpy as np
import monai.transforms as mt

def get_transforms(shape, norm_clip, pixdim):
    train_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS", lazy=True),
            mt.Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
                lazy=True
            ),
            mt.CropForegroundd(keys=["image", "label"], source_key="label", 
                               allow_smaller=True, lazy=True),
            mt.EnsureTyped(
                keys=["image", "label"], 
                dtype=[torch.float32, torch.long],
                track_meta=False),
            mt.RandAffined(
                keys=["image","label"],
                prob=1.0,
                spatial_size=shape,
                rotate_range=(np.pi/9, np.pi/9, np.pi/9),    # ±20°
                scale_range=(0.1,0.1,0.1),                   # ±10%
                mode=("bilinear","nearest"),
                padding_mode="border",
                lazy=True
            ),
            # mt.RandSpatialCropd(
            #     keys=["image", "label"], 
            #     roi_size=shape,
            #     lazy=True),
            # mt.Rand3DElasticd(
            #     keys=["image", "label"],
            #     prob=1.0,
            #     sigma_range=(3, 7),
            #     magnitude_range=(0.0, 10.0),
            #     rotate_range=(np.pi/9, np.pi/9, np.pi/9),
            #     shear_range=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
            #     translate_range=(5, 5, 5),
            #     scale_range=(0.1, 0.1, 0.1),
            #     mode=("bilinear", "nearest"),
            #     padding_mode="border"),
            mt.ScaleIntensityRanged(
                keys=["image"], 
                a_min=norm_clip[0],
                a_max=norm_clip[1],
                b_min=norm_clip[2],
                b_max=norm_clip[3],
                clip=True),
            mt.RandShiftIntensityd(
                keys=["image"],
                prob=0.50,
                offsets=0.20),
            mt.RandGaussianNoised(
                keys=["image"],
                prob=0.50,
                mean=0.0,
                std=0.10),
        ]
    )
    val_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS", lazy=True),
            mt.Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
                lazy=True
            ),
            mt.CropForegroundd(keys=["image", "label"], source_key="label", 
                               allow_smaller=True, lazy=True),
            mt.EnsureTyped(
                keys=["image", "label"], 
                dtype=[torch.float32, torch.long],
                track_meta=False),
        ]
    )
    return train_transform, val_transform


def get_data_files(
    images_dir: str,
    labels_dir: str,
    extension: str = ".nii.gz"
) -> List[Dict[str, str]]:
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
        {"image": str(images_dir / name), "label": str(labels_dir / name)}
        for name in image_names
    ]




if __name__ == "__main__":
    from monai.data import DataLoader, PersistentDataset
    from tqdm import tqdm
    torch.serialization.add_safe_globals([np.dtype, np.dtypes.Int64DType,
                                          np.ndarray, np.core.multiarray._reconstruct])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_clip = (-175, 250, -1.0, 1.0)
    pixdim = (1.0, 1.0, 1.0)
    shape = (128, 128, 128)

    # Deterministic transforms
    transforms, _ = get_transforms(
        shape=shape,
        norm_clip=norm_clip,
        pixdim=pixdim)
    
    # Instantiate datasets
    dataset = PersistentDataset(
        data = get_data_files(
            images_dir="data/FLARE-Task2-LaptopSeg/train_gt_label/imagesTr",
            labels_dir="data/FLARE-Task2-LaptopSeg/train_gt_label/labelsTr"),
        transform=transforms,
        cache_dir="data/cache/gt_label",
    )

    # Wrap in DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=64)

    # Iterate
    shapes = []
    for batch in tqdm(loader, desc="Processing batches"):
        img = batch["image"]; label = batch["label"]
        shapes.append(img.squeeze().shape)

    # Mean, Max, Min of each dimension
    shapes = torch.tensor(shapes)
    mean_shape = shapes.float().mean(dim=0)
    max_shape = shapes.float().max(dim=0).values
    min_shape = shapes.float().min(dim=0).values
    print("Shape statistics:")
    print(f"  Mean shape: {mean_shape.numpy()}")
    print(f"  Max shape: {max_shape.numpy()}")
    print(f"  Min shape: {min_shape.numpy()}")
