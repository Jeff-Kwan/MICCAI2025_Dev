from pathlib import Path
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset

import monai.transforms as mt


class SegDataset(Dataset):
    """
    PyTorch Dataset for 3D segmentation where both images and labels are stored
    as `.nii.gz` volumes. The transform should include MONAI keys
    (e.g. LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, etc.).

    Expects:
      - For each volume, `images_dir/<basename>.nii.gz`
      - For each corresponding mask, `labels_dir/<basename>.nii.gz`

    Returns (per __getitem__):
      a dict with at least the keys:
        {
          "image": <Tensor after transforms, shape = [C, D, H, W]>,
          "label": <Tensor after transforms, shape = [C, D, H, W]>
        }

    Args:
        images_dir (str or Path): Path to folder containing `<basename>.nii.gz` image volumes.
        labels_dir (str or Path): Path to folder containing `<basename>.nii.gz` segmentation masks.
        transform (callable, optional): MONAI-style transform to apply on each sample. It
            should expect and return a dict with at least the keys "image" and "label".
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform: Optional[mt.Transform] = None,
    ):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform

        # Verify directories exist
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.images_dir!r}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Label directory not found: {self.labels_dir!r}")

        # Gather all .nii.gz in images_dir
        image_paths = sorted(self.images_dir.glob("*.nii.gz"))
        if not image_paths:
            raise RuntimeError(f"No '.nii.gz' files found in {self.images_dir!r}")

        # Gather all .nii.gz in labels_dir and build a map basename → full path
        label_paths = sorted(self.labels_dir.glob("*.nii.gz"))
        if not label_paths:
            raise RuntimeError(f"No '.nii.gz' files found in {self.labels_dir!r}")

        # Build {basename: Path} map for labels
        # We strip exactly the suffix ".nii.gz"
        label_map = {}
        for lp in label_paths:
            name = lp.name
            if not name.lower().endswith(".nii.gz"):
                continue
            base = name[: -len(".nii.gz")]
            label_map[base] = lp

        # Match images ↔ labels
        self.examples: List[Dict[str, Path]] = []
        missing_labels: List[str] = []
        for img_path in image_paths:
            img_name = img_path.name
            if not img_name.lower().endswith(".nii.gz"):
                continue
            base = img_name.replace("_0000.", ".")[: -len(".nii.gz")]
            lbl_path = label_map.get(base)
            if lbl_path is None:
                # No matching label for this image
                missing_labels.append(img_name)
                continue
            # Store a dict of file paths (MONAI’s LoadImaged will read them later)
            self.examples.append({"image": str(img_path), "label": str(lbl_path)})

        if not self.examples:
            raise RuntimeError(
                f"No matching label '.nii.gz' found for any image in {self.images_dir!r}")

        if missing_labels:
            print("Warning: The following images had no matching label and were skipped:")
            for fn in missing_labels:
                print(f"  • {fn}")
            raise RuntimeWarning(
                f"Some images had no matching label: {len(missing_labels)} out of {len(image_paths)}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.examples[idx].copy()  # {"image": "<path>", "label": "<path>"}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    # Reference: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb
    from monai.data import DataLoader
    import monai.transforms as mt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (96, 96, 96)
    train_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            mt.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"),
            mt.Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            mt.EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            mt.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=shape,
                pos=1,
                neg=1,
                image_key="image",
                image_threshold=0,
            ),
            mt.RandAxisFlip(
                prob=0.10,
            ),
            mt.RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            mt.RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            # mt.RandAffined(
            #     keys=["image", "label"],
            #     mode=("bilinear", "nearest"),
            #     prob=0.5,
            #     rotate_range=(0.1, 0.1, 0.1),
            #     scale_range=(0.2, 0.2, 0.2),
            #     padding_mode="zeros"),
            mt.RandGaussianNoised(
                keys=["image"],
                prob=0.1,
                mean=0.0,
                std=0.05,
            ),
        ]
    )
    val_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            mt.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"),
            mt.Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            mt.EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )

    # Instantiate datasets
    train_ds = SegDataset(
        images_dir="data/FLARE-Task2-LaptopSeg/train_pseudo_label/imagesTr",
        labels_dir="data/FLARE-Task2-LaptopSeg/train_pseudo_label/flare22_aladdin5_pseudo",
        transform=train_transform,
    )
    # val_ds = SegDataset(
    #     images_dir="data/FLARE-Task2-LaptopSeg/train_gt_label/imagesTr",
    #     labels_dir="data/FLARE-Task2-LaptopSeg/train_gt_label/labelsTr",
    #     transform=val_transform,
    # )

    # Wrap in DataLoader
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)

    # Iterate
    for batch in train_loader:
        img = batch["image"]; label = batch["label"]
        print(img.shape, label.shape)
        print(img.dtype, label.dtype)
        exit()
