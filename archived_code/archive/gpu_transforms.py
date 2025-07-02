import torch
import numpy as np
from monai import transforms as mt

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# dummy data dict
shape = (64, 64, 64)       # or whatever your roi/crop/etc size is
num_crops = 2
dummy_image = torch.randn(1, *shape, device=device)  # 1-channel
dummy_label = torch.randint(0, 14, (1, *shape), device=device)

base_data = {"image": dummy_image, "label": dummy_label}

# your list of transforms
transforms = [
    mt.CropForegroundd(
                keys=["image"],
                source_key="label"),
    mt.RandSpatialCropSamplesd(
        keys=["image", "label"],
        roi_size=shape,
        num_samples=num_crops,
        lazy=True,
    ),
    mt.RandAffined(
        keys=["image","label"],
        prob=1.0,
        spatial_size=shape,
        rotate_range=(np.pi/6, np.pi/6, np.pi/6),
        scale_range=(0.1, 0.1, 0.1),
        mode=("bilinear","nearest"),
        padding_mode="border",
        lazy=True,
    ),
    mt.RandFlipd(
        keys=["image", "label"],
        prob=1.0,
        spatial_axis=(0, 1),
        lazy=True,
    ),
    mt.RandRotate90d(
        keys=["image", "label"],
        prob=1.0,
        spatial_axes=(0, 1),
        lazy=True,
    ),
    mt.Rand3DElasticd(
        keys=["image", "label"],
        prob=1.0,
        sigma_range=(2.0, 5.0),
        magnitude_range=(1.0, 3.0),
        spatial_size=shape,
        rotate_range=(np.pi/9, np.pi/9, np.pi/9),
        scale_range=(0.1, 0.1, 0.1),
        shear_range=(0.0, 0.0, 0.0),
        mode=("bilinear", "nearest"),
    ),
    mt.SpatialPadd(
        keys=["image", "label"],
        spatial_size=shape,
        mode=("edge", "edge"),
        lazy=True,
    ),
    mt.RandGaussianSmoothd(keys="image", prob=1.0),
    mt.RandGaussianNoised(keys="image", prob=1.0),
    mt.RandBiasFieldd(keys="image", prob=1.0),
    mt.RandAdjustContrastd(keys="image", prob=1.0),
    mt.RandGaussianSharpend(keys="image", prob=1.0),
    mt.RandHistogramShiftd(keys="image", prob=1.0),
    mt.RandCoarseDropoutd(
        keys=["image"],
        prob=1.0,
        holes=2,
        max_holes=4,
        spatial_size=(24, 24, 24),
        max_spatial_size=(48, 48, 48),
    ),
    mt.RandCoarseShuffled(
        keys=["image"],
        prob=1.0,
        holes=4,
        max_holes=8,
        spatial_size=(8, 8, 8),
        max_spatial_size=(24, 24, 24),
    ),
]

# test each transform
for t in transforms:
    name = t.__class__.__name__
    # move to GPU if possible
    try:
        t.to(device)
    except Exception:
        # some transforms don't implement .to(), ignore
        pass

    data = {k: v.clone() for k, v in base_data.items()}
    print(f"\n=== Testing {name} ===")
    try:
        out = t(data)
        # for lazy transforms MONAI may return a dict of lazy objects,
        # you can call apply_transforms to force execution:
        if hasattr(t, "apply_transforms"):
            out = t.apply_transforms(data)
    except Exception as e:
        print(f"  ✗ Raised exception: {e}")
        continue

    if isinstance(out, dict):
        img = out.get("image")
        label = out.get("label")
        if img is not None and label is not None:
            if img.device.type == 'cuda' and label.device.type == 'cuda':
                continue
    print(f"  ⚠️  {name} does NOT support GPU")