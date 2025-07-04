import os
from pathlib import Path
from typing import List, Dict
import torch
import numpy as np
import monai.transforms as mt

class SafeCropForegroundd(mt.MapTransform):
    def __init__(self, keys, source_key, shape, margin=0):
        super().__init__(keys)
        self.source_key = source_key
        self.cropforeground = mt.CropForegroundd(keys=keys, source_key=source_key,
                                    margin=margin, allow_smaller=True, lazy=True)
        self.randcrop = mt.RandSpatialCropd(keys=keys, roi_size=shape, lazy=True)
        

    def __call__(self, data):
        if data.get(self.source_key).any():  # Check if mask has any non-zero values
            self.cropforeground(data)
        return self.randcrop(data)

def get_transforms(shape, spatial, intensity, coarse, soft=False):
    label_dtype = torch.uint8 if soft else torch.long
    train_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.RandSpatialCropd(
                keys=["image", "label"], 
                roi_size=shape,
                lazy=True),
            mt.DivisiblePadd(
                keys=["image", "label"],
                k=16,
                lazy=True),
            mt.OneOf(       # Random spatial augmentations
                transforms=[
                    mt.Identityd(keys=["image", "label"]),
                    mt.RandAffined(     # Small affine perturbation
                        keys=["image","label"],
                        prob=1.0,
                        spatial_size=shape,
                        rotate_range=(np.pi/9, np.pi/9, np.pi/9),
                        scale_range=(0.1, 0.1, 0.1),
                        translate_range=(4, 4, 4),
                        mode=("trilinear", "nearest"),
                        padding_mode="border",
                        lazy=True),
                    mt.RandFlipd(
                        keys=["image", "label"],
                        prob=1.0,
                        spatial_axis=(0, 1),
                        lazy=True),  # Flip in XY plane
                    mt.RandRotate90d(
                        keys=["image", "label"],
                        prob=1.0,
                        spatial_axes=(0, 1),
                        lazy=True),  # Rotate in XY plane
                    mt.Rand3DElasticd(
                        keys=["image", "label"],
                        prob=1.0,
                        sigma_range=(2.0, 5.0),
                        magnitude_range=(1.0, 3.0),
                        spatial_size=shape,
                        translate_range=(4, 4, 4),
                        rotate_range=(np.pi/9, np.pi/9, np.pi/9),  # ±20°
                        scale_range=(0.1, 0.1, 0.1),                # ±10%
                        mode=("trilinear", "nearest")
                    )],
                weights=spatial),
            mt.OneOf(     # Random intensity augmentations
                transforms=[
                    mt.Identityd(keys=["image"]),
                    mt.RandGaussianSmoothd(keys='image', prob=1.0),
                    mt.RandGaussianNoised(keys='image', prob=1.0),
                    mt.RandBiasFieldd(keys='image', prob=1.0),
                    mt.RandAdjustContrastd(keys='image', prob=1.0),
                    mt.RandGaussianSharpend(keys='image', prob=1.0),
                    mt.RandHistogramShiftd(keys='image', prob=1.0)],
                weights=intensity),
            mt.OneOf(   # Random coarse augmentations
                transforms=[
                    mt.Identityd(keys=["image"]),
                    mt.RandCoarseDropoutd(
                        keys=["image"],
                        prob=1.0,
                        holes=1,
                        max_holes=4,
                        spatial_size=(16, 16, 16),
                        max_spatial_size=(32, 32, 32)),
                    mt.RandCoarseShuffled(
                        keys=["image"],
                        prob=1.0,
                        holes=8, max_holes=16,
                        spatial_size=(6, 6, 6),
                        max_spatial_size=(12, 12, 12))],
                weights=coarse),
            mt.EnsureTyped(
                keys=["image", "label"], 
                dtype=[torch.float32, label_dtype],
                track_meta=False),
        ]
    )
    val_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.CropForegroundd(
                keys=["image", "label"],
                source_key="label",
                margin=48,
                allow_smaller=True),
            mt.DivisiblePadd(
                keys=["image", "label"],
                k=16,
                lazy=True),
            mt.EnsureTyped(
                keys=["image", "label"], 
                dtype=[torch.float32, torch.long],
                track_meta=False),
        ]
    )
    return train_transform, val_transform


def get_vae_transforms(shape, spatial, intensity, coarse):
    train_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.RandSpatialCropd(
                keys=["image", "label"], 
                roi_size=shape,
                lazy=True),
            mt.OneOf(
                transforms=[
                    mt.OneOf(       # Random spatial augmentations
                        transforms=[
                            mt.Identityd(keys=["image", "label"]),
                            mt.RandAffined(     # Small affine perturbation
                                keys=["image","label"],
                                prob=1.0,
                                spatial_size=shape,
                                translate_range=(8, 8, 8),
                                rotate_range=(np.pi/9, np.pi/9, np.pi/9),
                                scale_range=(0.1, 0.1, 0.1),
                                mode=("trilinear", "nearest"),
                                padding_mode="border",
                                lazy=True),
                            mt.RandFlipd(
                                keys=["image", "label"],
                                prob=1.0,
                                spatial_axis=(0, 1),
                                lazy=True),  # Flip in XY plane
                            mt.RandRotate90d(
                                keys=["image", "label"],
                                prob=1.0,
                                spatial_axes=(0, 1),
                                lazy=True),  # Rotate in XY plane
                            mt.Rand3DElasticd(
                                keys=["image", "label"],
                                prob=1.0,
                                sigma_range=(2.0, 5.0),
                                magnitude_range=(1.0, 3.0),
                                spatial_size=shape,
                                translate_range=(8, 8, 8),
                                rotate_range=(np.pi/9, np.pi/9, np.pi/9),  # ±20°
                                scale_range=(0.1, 0.1, 0.1),                # ±10%
                                mode=("trilinear", "nearest")
                            )],
                        weights=spatial),
                    mt.OneOf(     # Random intensity augmentations
                        transforms=[
                            mt.Identityd(keys=["image"]),
                            mt.RandGaussianSmoothd(keys='image', prob=1.0),
                            mt.RandGaussianNoised(keys='image', prob=1.0),
                            mt.RandBiasFieldd(keys='image', prob=1.0),
                            mt.RandAdjustContrastd(keys='image', prob=1.0),
                            mt.RandGaussianSharpend(keys='image', prob=1.0),
                            mt.RandHistogramShiftd(keys='image', prob=1.0)],
                        weights=intensity),
                    mt.OneOf(   # Random coarse augmentations
                        transforms=[
                            mt.Identityd(keys=["image"]),
                            mt.RandCoarseDropoutd(
                                keys=["image"],
                                prob=1.0,
                                holes=1,
                                max_holes=4,
                                spatial_size=(16, 16, 16),
                                max_spatial_size=(32, 32, 32)),
                            mt.RandCoarseShuffled(
                                keys=["image"],
                                prob=1.0,
                                holes=8, max_holes=16,
                                spatial_size=(6, 6, 6),
                                max_spatial_size=(12, 12, 12))],
                        weights=coarse)],
                weights=[2, 1, 1]
            ),
            mt.DivisiblePadd(
                keys=["image", "label"],
                k=16),
            mt.EnsureTyped(
                keys=["image", "label"], 
                dtype=[torch.float32, torch.long],
                track_meta=False),
        ]
    )
    val_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.CropForegroundd(
                keys=["image", "label"],
                source_key="label",
                margin=4,
                allow_smaller=True),
            mt.DivisiblePadd(
                keys=["image", "label"],
                k=16),
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


def get_mim_data_files(
    images_dir: str,
    extension: str = ".nii.gz"
    ) -> List[Dict[str, str]]:
    """
    Returns a list of dicts with file paths for images.
    Each dict has the key "image".

    Raises:
        FileNotFoundError: if the directory does not exist.
        RuntimeError: if no files with the given extension are found.
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
    return [{"image": str(images_dir / name)} for name in image_names]


if __name__ == "__main__":
    from monai.data import DataLoader, PersistentDataset
    from tqdm import tqdm
    torch.serialization.add_safe_globals([np.dtype, np.dtypes.Int64DType,
                                          np.ndarray, np.core.multiarray._reconstruct])
    device = torch.device("cuda")
    shape = (128, 128, 128)

    # Deterministic transforms
    transforms, _ = get_transforms(
        shape=shape,
        num_crops=8,
        device=device)
    
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
