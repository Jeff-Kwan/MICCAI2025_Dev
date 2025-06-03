import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool
from tqdm import tqdm


class FLARESegDataset(Dataset):
    """
    PyTorch Dataset for FLARE segmentation (loads NIfTI .nii.gz pairs).
    """
    def __init__(self, img_dir: str, label_dir: str, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # collect all *.nii.gz in img_dir
        self.image_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.nii.gz")))
        self.label_paths = []
        for img_path in self.image_paths:
            # remove the modality suffix ("_0000") to match the label filename
            base = os.path.basename(img_path).replace('_0000', '')
            label_path = os.path.join(self.label_dir, base)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label not found for {img_path}")
            self.label_paths.append(label_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
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
            "filename": os.path.basename(img_path)  # e.g. "case123_0000.nii.gz"
        }
        return sample


def _process_and_save(args):
    """
    Worker function for a single sample. Loads NIfTI, applies transform,
    converts to NumPy, and saves image, label, and spacing arrays.
    """
    img_path, lbl_path, img_out_dir, lbl_out_dir, spacing_out_dir, transform = args

    # Load NIfTI volumes
    img_nib = nib.load(img_path, mmap=True)
    lbl_nib = nib.load(lbl_path, mmap=True)

    # Convert to torch Tensors
    image = torch.from_numpy(np.asarray(img_nib.dataobj, dtype=np.float32))
    label = torch.from_numpy(np.asarray(lbl_nib.dataobj, dtype=np.int16)).long()

    # Apply transform if provided
    if transform is not None:
        image, label = transform((image, label))

    # Convert back to NumPy
    image_np = image.numpy()
    label_np = label.numpy()

    # Extract and save spacing (tuple of floats)
    spacing = np.array(img_nib.header.get_zooms(), dtype=np.float32)

    # Derive base name without "_0000.nii.gz"
    filename = os.path.basename(img_path)
    base = filename.replace("_0000.nii.gz", "")

    # Build save paths
    img_save_path = os.path.join(img_out_dir, base + "_image.npy")
    lbl_save_path = os.path.join(lbl_out_dir, base + "_label.npy")
    spacing_save_path = os.path.join(spacing_out_dir, base + "_spacing.npy")

    # Actually write the .npy files
    np.save(img_save_path, image_np)
    np.save(lbl_save_path, label_np)
    np.save(spacing_save_path, spacing)

    # Return a status message for logging
    return (
        f"Saved:\n"
        f"    Image:   {img_save_path}\n"
        f"    Label:   {lbl_save_path}\n"
        f"    Spacing: {spacing_save_path}"
    )


def convert_to_npy(
    img_dir: str,
    label_dir: str,
    output_root: str,
    transform=None
):
    """
    Loads all image/label pairs via FLARESegDataset and saves them as .npy files
    using multiple CPU cores in parallel.

    Args:
        img_dir (str): path to folder containing image .nii.gz files (e.g. ".../imagesTr").
        label_dir (str): path to folder containing label .nii.gz files (e.g. ".../labelsTr").
        output_root (str): output root directory, which will contain three subfolders:
                              output_root/images/
                              output_root/labels/
                              output_root/spacings/
        transform (callable, optional): any preprocessing to apply in each worker.
    """
    # Create dataset (just to gather paths; we won't iterate in a single‐threaded loop)
    dataset = FLARESegDataset(img_dir=img_dir, label_dir=label_dir, transform=transform)

    # Prepare output directories
    img_out_dir = os.path.join(output_root, "images")
    lbl_out_dir = os.path.join(output_root, "labels")
    spacing_out_dir = os.path.join(output_root, "spacings")

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)
    os.makedirs(spacing_out_dir, exist_ok=True)

    num_samples = len(dataset)
    print(f"Found {num_samples} samples. Saving to:\n"
          f"  {img_out_dir}\n"
          f"  {lbl_out_dir}\n"
          f"  {spacing_out_dir}\n")

    # Build an argument list for each sample
    args_list = []
    for img_path, lbl_path in zip(dataset.image_paths, dataset.label_paths):
        args_list.append(
            (
                img_path,
                lbl_path,
                img_out_dir,
                lbl_out_dir,
                spacing_out_dir,
                transform
            )
        )

    # Launch a process pool with as many workers as there are CPU cores
    n_workers = os.cpu_count() or 1
    with Pool(processes=n_workers) as pool:
        # use imap so we can wrap in tqdm
        results = []
        for msg in tqdm(pool.imap(_process_and_save, args_list),
                        total=num_samples,
                        desc="Converting",
                        ncols=80):
            results.append(msg)

    # Print a unified, ordered log
    for idx, msg in enumerate(results, start=1):
        print(f"[{idx:03d}/{num_samples:03d}] {msg}\n")

    print("All files saved as .npy.")


if __name__ == "__main__":
    #
    # ==== USER CONFIGURATION SECTION ====
    #
    # Edit these three paths as needed:
    #
    img_dir = "./data/FLARE-Task2-LaptopSeg/train_pseudo_label/imagesTr"
    label_dir = "./data/FLARE-Task2-LaptopSeg/train_pseudo_label/flare22_aladdin5_pseudo"
    # Where to put the .npy outputs. Three subfolders (images/, labels/, spacings/) will be created under this path.
    output_root = "./data/FLARE-Task2-LaptopSeg/data_npy"

    # If you have any transforms (e.g. cropping, normalization), pass them here; otherwise set to None.
    my_transform = None
    #
    # ====================================
    #

    convert_to_npy(
        img_dir=img_dir,
        label_dir=label_dir,
        output_root=output_root,
        transform=my_transform
    )
