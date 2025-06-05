import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class FLARESegDataset_nz(Dataset):
    """
    """
    def __init__(self, img_dir: str, label_dir: str, transform=None):
        """
        """
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # collect all *.nii.gz in img_dir
        self.image_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.nii.gz")))
        self.label_paths = []
        for img_path in self.image_paths:
            base = os.path.basename(img_path).replace('_0000', '')
            label_path = os.path.join(self.label_dir, base)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label not found for {img_path}")
            self.label_paths.append(label_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        """
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]

        # Load image and label with nibabel (H, W, D)
        img_nib = nib.load(img_path, mmap=True)
        lbl_nib = nib.load(lbl_path, mmap=True)
        image = torch.from_numpy(np.asarray(img_nib.dataobj, dtype=np.float32))
        label = torch.from_numpy(np.asarray(lbl_nib.dataobj, dtype=np.int16)).long()

        if self.transform is not None:
            image, label = self.transform((image, label))

        sample = {
            "image": image,
            "label": label,
            "spacing": img_nib.header.get_zooms(),
            "filename": os.path.basename(img_path)
        }

        return sample


class FLARESegDataset_npy(Dataset):
    """
    """
    def __init__(self, img_dir: str, label_dir: str, spacing_dir: str, transform=None):
        """
        """
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.spacing_dir = spacing_dir
        self.transform = transform

        # collect all *.nii.gz in img_dir
        self.image_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.npy")))
        self.label_paths = []
        self.spacing_paths = []
        for img_path in self.image_paths:
            base = os.path.basename(img_path)
            label_path = os.path.join(self.label_dir, base.replace('image', 'label'))
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label not found for {img_path}")
            self.label_paths.append(label_path)
            spacing_path = os.path.join(self.spacing_dir, base.replace('image', 'spacing'))
            if not os.path.exists(spacing_path):
                raise FileNotFoundError(f"spacing not found for {img_path}")
            self.spacing_paths.append(spacing_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]
        spacing_path = self.spacing_paths[idx]

        # Load image and label with nibabel (H, W, D)
        image = torch.from_numpy(np.load(img_path))
        label = torch.from_numpy(np.load(lbl_path)).long()
        spacing = np.load(spacing_path)

        if self.transform is not None:
            image, label = self.transform((image, label))

        sample = {
            "image": image,
            "label": label,
            "spacing": spacing,
            "filename": os.path.basename(img_path)
        }
        return sample

if __name__ == '__main__':
    # Example usage
    img_dir = "./data/FLARE-Task2-LaptopSeg/train_gt_label/imagesTr"
    label_dir = "./data/FLARE-Task2-LaptopSeg/train_gt_label/labelsTr"

    dataset = FLARESegDataset_nz(img_dir=img_dir, label_dir=label_dir)
    sample = dataset[0]
    print(f"Sample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Label shape: {sample['label'].shape}")
    print(f"  Spacing: {sample['spacing']}")
    print(f"  Filename: {sample['filename']}")
    print("-" * 40)
