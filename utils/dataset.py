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


def get_data_files(images_dir: str, labels_dir: str) -> List[Dict[str, str]]:
    """
    Returns a list of dicts with file paths for images and labels.
    Each dict has the keys "image" and "label".
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    transform = transform

    # Verify directories exist
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir!r}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels_dir!r}")

    # Gather all .nii.gz in images_dir
    image_paths = sorted(images_dir.glob("*.nii.gz"))
    if not image_paths:
        raise RuntimeError(f"No '.nii.gz' files found in {images_dir!r}")

    # Gather all .nii.gz in labels_dir and build a map basename → full path
    label_paths = sorted(labels_dir.glob("*.nii.gz"))
    if not label_paths:
        raise RuntimeError(f"No '.nii.gz' files found in {labels_dir!r}")
    
    # Build data files dictionary-list
    data_files = []
    for img_path in image_paths:
        img_name = img_path.name
        lbl_path = labels_dir / img_name
        if not lbl_path.is_file():
            raise ValueError(
                f"Missing label for image {img_name!r}: expected {lbl_path!r} to exist"
            )
        data_files.append({"image": str(img_path), "label": str(lbl_path)})
    return data_files




if __name__ == "__main__":
    # Reference: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb
    from monai.data import DataLoader, PersistentDataset
    import monai.transforms as mt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (96, 96, 96)
    norm_clip = (-175, 250, 0.0, 1.0)
    pixdim = (1.5, 1.5, 2.0)

    # Deterministic transforms
    val_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.ScaleIntensityRanged(keys=["image"], 
                                    a_min=norm_clip[0],
                                    a_max=norm_clip[1],
                                    b_min=norm_clip[2],
                                    b_max=norm_clip[3],
                                    clip=True),
            mt.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"),
            mt.Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            mt.EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )

    # Instantiate datasets
    val_ds = PersistentDataset(
        data = get_data_files(
            images_dir="data/FLARE-Task2-LaptopSeg/train_gt_label/imagesTr",
            labels_dir="data/FLARE-Task2-LaptopSeg/train_gt_label/labelsTr"),
        transform=val_transform,
        cache_dir="data/cache/gt_label",
    )

    # Wrap in DataLoader
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)

    # Iterate
    shapes = []
    for batch in val_loader:
        img = batch["image"]; label = batch["label"]
        shapes.append(img.shape)
        print(img.shape, label.shape)
        print(img.dtype, label.dtype)

    # Mean, Max, Min of each dimension
    shapes = torch.tensor(shapes)
    mean_shape = shapes.float().mean(dim=0)
    max_shape = shapes.float().max(dim=0).values
    min_shape = shapes.float().min(dim=0).values
    print("Shape statistics:")
    print(f"  Mean shape: {mean_shape.numpy()}")
    print(f"  Max shape: {max_shape.numpy()}")
    print(f"  Min shape: {min_shape.numpy()}")
