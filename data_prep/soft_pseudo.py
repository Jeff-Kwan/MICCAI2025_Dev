import os
from pathlib import Path
import monai.transforms as mt
from monai.data import Dataset, DataLoader
from monai.networks.utils import one_hot
import torch
from torch.nn.functional import normalize


def get_data_files(dir1, dir2, extension = ".nii.gz"):
    """
    Returns a list of dicts with file paths for images and labels.
    Each dict has the keys "image" and "label".

    Raises:
        FileNotFoundError: if either directory does not exist.
        RuntimeError: if no files with the given extension are found.
        ValueError: if any image is missing a matching label.
    """
    dir1 = Path(dir1)
    dir2 = Path(dir2)

    # Scan image directory
    labels_1_names = sorted(
        entry.name
        for entry in os.scandir(dir1)
        if entry.is_file() and entry.name.endswith(extension))

    # Scan label directory once, build a set of names
    labels_2_names = sorted(
        entry.name
        for entry in os.scandir(dir2)
        if entry.is_file() and entry.name.endswith(extension))

    # Detect any missing labels in one go
    missing = [name for name in labels_1_names if name not in labels_2_names]
    if missing:
        missing_list = ", ".join(repr(n) for n in missing)
        raise ValueError(f"Missing labels correspondence: {missing_list}")

    # Build result list
    return [
        {"lbl1": str(dir1 / name), "lbl2": str(dir2 / name), 
         "base_name": name.removesuffix(".nii.gz")}
        for name in labels_1_names
    ]


def create_soft_pseudo(dir1, dir2, soft_labels_dir, weights, extension = ".nii.gz"):
    data_files = get_data_files(dir1, dir2, extension)

    load_transform = mt.Compose(
        mt.LoadImaged(["lbl1", "lbl2"]),
        mt.EnsureTyped(["lbl1", "lbl2"], dtype=torch.long, track_meta=True))
    saver = mt.SaveImaged(
        keys=["soft"],
        meta_keys=["lbl1_meta_dict"],
        output_dir=soft_labels_dir,
        output_postfix="",
        output_ext=".nii.gz",
        separate_folder=False,
        output_dtype=torch.float32,
        print_log=False)
    dataset = Dataset(data=data_files, transform=load_transform)

    for data in dataset:
        lbl1 = one_hot(data["lbl1"][0].permute(3, 0, 1, 2)).float()
        lbl2 = one_hot(data["lbl2"][0].permute(3, 0, 1, 2)).float()

        # Create soft pseudo-label
        data["soft"] = normalize(weights[0] * lbl1 + weights[1] * lbl2, p=1, dim=0)

        # Save the soft pseudo-label
        saver(data)

if __name__ == "__main__":
    aladdin5 = "data/nifti/train_pseudo/aladdin5"
    blackbean = "data/nifti/train_pseudo/blackbean"
    soft_labels = "data/nifti/train_pseudo/soft_labels"

    weights = [0.6, 0.4]    # Trust Aladdin5 labels more
    os.makedirs(soft_labels, exist_ok=True)
    create_soft_pseudo(aladdin5, blackbean, soft_labels, weights, extension=".nii.gz")