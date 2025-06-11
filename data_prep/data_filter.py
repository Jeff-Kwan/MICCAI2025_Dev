import os
from pathlib import Path
from tqdm import tqdm
import torch
from monai.transforms import LoadImaged, EnsureTyped, Compose
from monai.data import Dataset, ThreadDataLoader
import multiprocessing

def get_data_files(images_dir, labels_dir, extension=".nii.gz"):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir!r}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels_dir!r}")

    image_paths = sorted(images_dir.glob(f"*{extension}"))
    if not image_paths:
        raise RuntimeError(f"No '{extension}' files found in {images_dir!r}")

    data_files = []
    for img_path in image_paths:
        lbl_path = labels_dir / img_path.name
        if not lbl_path.exists():
            raise ValueError(f"Missing label for image: {img_path.name!r}")
        # strip off the full extension:
        base = img_path.name[:-len(extension)]
        data_files.append({
            "image": str(img_path),
            "label": str(lbl_path),
            "base_name": base
        })
    return data_files

def filter_and_delete_zeros(data_files, num_workers=None):
    num_workers = num_workers or min(32, multiprocessing.cpu_count())
    # only load the label, leave base_name untouched
    transform = Compose([
        LoadImaged(keys=["label"]),
        EnsureTyped(keys=["label"], dtype=torch.uint8, track_meta=True),
    ])

    ds = Dataset(data=data_files, transform=transform)
    dl = ThreadDataLoader(ds, batch_size=1, num_workers=num_workers)

    # build a quick lookup from base_name → entry
    file_map = {d["base_name"]: d for d in data_files}

    for batch in tqdm(dl, desc="Filtering zero‐label images"):
        lbl = batch["label"][0].numpy().squeeze()
        # fast “all zeros” check
        if not lbl.any():
            base = batch["base_name"][0]
            entry = file_map[base]
            img_p = Path(entry["image"])
            lbl_p = Path(entry["label"])
            print(f"Deleting all‐zero label pair:\n  Image: {img_p}\n  Label: {lbl_p}")
            for p in (img_p, lbl_p):
                try:
                    p.unlink()
                except Exception as e:
                    print(f"  ✖ Couldn’t delete {p}: {e}")

if __name__ == "__main__":
    splits = [
        ("data/preprocessed/val/images",   "data/preprocessed/val/labels"),
        ("data/preprocessed/train_gt/images",    "data/preprocessed/train_gt/labels"),
        ("data/preprocessed/train_pseudo/images","data/preprocessed/train_pseudo/aladdin5"),
    ]
    all_files = []
    for img_dir, lbl_dir in splits:
        all_files.extend(get_data_files(img_dir, lbl_dir))

    filter_and_delete_zeros(all_files)
