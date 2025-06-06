from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CTSegmentationDataset(Dataset):
    """
    PyTorch Dataset for 3D organ CT segmentation (with pre-converted .npy volumes).

    Expects:
      - For each volume, `images_dir/<basename>.npy` exists
      - For each corresponding mask, `labels_dir/<basename>.npy` exists
      - For each image, an optional `images_dir/<basename>_metadata.json` exists

    Returns, for each __getitem__(idx):
      {
        "image":    Tensor(float32)  shape = [1, D, H, W],
        "label":    Tensor(int64)    shape = [1, D, H, W],
        "metadata": dict or None     minimal header info from JSON (if found)
      }

    Args:
        images_dir (str or Path): Path to folder containing `<basename>.npy` image volumes.
        labels_dir (str or Path): Path to folder containing `<basename>.npy` segmentation masks.
        transform (callable, optional): Function to apply to each sample dict
                                        (e.g. random crop, flip, intensity norm, etc.).
    """

    def __init__(self, images_dir, labels_dir, transform=None):
        super().__init__()

        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform

        # Sanity check: both directories must exist
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.images_dir!r}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Label directory not found: {self.labels_dir!r}")

        # Collect only image files ending with '.npy'
        image_paths = sorted(self.images_dir.glob("*.npy"))
        if not image_paths:
            raise RuntimeError(f"No image files matching '*.npy' in {self.images_dir!r}")

        # Build a map from label stem → label path
        label_paths = sorted(self.labels_dir.glob("*.npy"))
        if not label_paths:
            raise RuntimeError(f"No label .npy files found in {self.labels_dir!r}")
        label_map = {p.stem: p for p in label_paths}

        # Match images ↔ labels
        self.examples = []
        missing = []

        for img_path in image_paths:
            # strip the trailing '' to get the base key
            # e.g. "liver_001" → "liver_001"
            stem = img_path.stem
            key = stem.rsplit("_", 1)[0]

            lbl_path = label_map.get(key)
            if lbl_path is None:
                missing.append(img_path.name)
                continue

            # optional metadata JSON: "<basename>_metadata.json"
            md_file = self.images_dir / f"{stem}_metadata.json"
            md_path = md_file if md_file.exists() else None

            self.examples.append((img_path, lbl_path, md_path))

        if not self.examples:
            raise RuntimeError(
                f"No matching label .npy found for any image in {self.images_dir!r}."
            )

        if missing:
            print("Warning: the following images have no matching label and will be skipped:")
            for fn in missing:
                print(f"  • {fn}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Returns:
            sample (dict):
                {
                  "image":    torch.FloatTensor, shape = [1, D, H, W],
                  "label":    torch.LongTensor, shape = [1, D, H, W],
                  "metadata": dict or None
                }
        """
        img_path, lbl_path, md_path = self.examples[idx]

        # Load image volume (assumed float or uint8, cast to float32)
        image_np = np.load(str(img_path))  # shape = [D, H, W]
        if image_np.ndim != 3:
            raise ValueError(
                f"Expected a 3D array at {img_path!r}, but got shape {image_np.shape}"
            )

        # Load label volume (assumed integer mask; cast to int64)
        label_np = np.load(str(lbl_path))  # shape = [D, H, W]
        if label_np.ndim != 3:
            raise ValueError(
                f"Expected a 3D array at {lbl_path!r}, but got shape {label_np.shape}"
            )

        # Add channel dimension: [D, H, W] → [1, D, H, W]
        image_np = np.expand_dims(image_np, axis=0)
        label_np = np.expand_dims(label_np, axis=0)

        # Convert to torch.Tensor
        image_tensor = torch.from_numpy(image_np).float()
        label_tensor = torch.from_numpy(label_np).long()

        # Load metadata JSON if it exists
        metadata = None
        if md_path is not None:
            try:
                with open(md_path, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Warning: failed to read metadata '{md_path.name}': {e}")
                metadata = None

        sample = {
            "image":    image_tensor,
            "label":    label_tensor,
            "metadata": metadata,
        }

        # Apply transform if provided (expects and returns a dict with "image" and "label" at minimum)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    images_folder = "./data/FLARE-Task2-LaptopSeg/train_gt_label/imagesTr"
    labels_folder = "./data/FLARE-Task2-LaptopSeg/train_gt_label/labelsTr"

    # Example transform: normalize each volume to zero mean / unit std
    def example_transform(sample):
        img, lbl = sample["image"], sample["label"]
        img = (img - img.mean()) / (img.std() + 1e-8)
        return {"image": img, "label": lbl, "metadata": sample["metadata"]}

    # Instantiate dataset
    dataset = CTSegmentationDataset(
        images_dir=images_folder,
        labels_dir=labels_folder,
        transform=example_transform,  # or None if you handle transforms elsewhere
    )

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,      # adjust to your CPU
    )

    # Iterate
    for batch in loader:
        imgs = batch["image"]       # shape = [B, 1, D, H, W]
        masks = batch["label"]      # shape = [B, 1, D, H, W]
        metas = batch["metadata"]   # list of length B (each element is a dict or None)

        print(f"Batch size: {imgs.shape[0]}")
        print(f"Image shape: {imgs.shape}")
        print(f"Label shape: {masks.shape}")
        print(f"Metadata: {metas}")
        break
