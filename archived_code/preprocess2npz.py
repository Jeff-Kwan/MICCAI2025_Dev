import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader
import multiprocessing as mp

from monai.transforms import MapTransform
from typing import Hashable, Sequence, Union

class MapLabelsToZeroOutsideRange(MapTransform):
    def __init__(
        self,
        keys: Union[Sequence[Hashable], Hashable],
        valid_labels: Sequence[int],
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.valid_labels = set(valid_labels)

    def __call__(self, data):
        for key in self.keys:
            invalid_mask = ~np.isin(data[key], list(self.valid_labels))
            data[key][invalid_mask] = 0
        return data

def get_data_files(images_dir, labels_dir, extension=".nii.gz"):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir!r}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels_dir!r}")

    image_names = sorted(
        entry.name
        for entry in os.scandir(images_dir)
        if entry.is_file() and entry.name.endswith(extension)
    )
    if not image_names:
        raise RuntimeError(f"No '{extension}' files found in {images_dir!r}")

    label_names = {
        entry.name
        for entry in os.scandir(labels_dir)
        if entry.is_file() and entry.name.endswith(extension)
    }
    if not label_names:
        raise RuntimeError(f"No '{extension}' files found in {labels_dir!r}")

    missing = [name for name in image_names if name not in label_names]
    if missing:
        missing_list = ", ".join(repr(n) for n in missing)
        raise ValueError(f"Missing labels for images: {missing_list}")

    return [
        {
            "image": str(images_dir / name),
            "label": str(labels_dir / name),
            "base_name": name.removesuffix(extension),
        }
        for name in image_names
    ]

def main(images_dir, labels_dir, out_image_dir, out_label_dir, pixdim, norm_clip):
    # create output dirs
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    # define the processing pipeline (no SaveImaged)
    transform = mt.Compose([
        mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        mt.Orientationd(keys=["image", "label"], axcodes="RAS", lazy=True),
        mt.SqueezeDimd(keys=["image", "label"], dim=0),
        mt.Spacingd(
            keys=["image", "label"],
            pixdim=pixdim,
            mode=("bicubic", "nearest"),
            lazy=True,
        ),
        mt.EnsureTyped(
            keys=["image", "label"],
            dtype=[torch.float32, torch.uint8],
            track_meta=True,
        ),
        mt.ScaleIntensityRanged(
            keys=["image"],
            a_min=norm_clip[0],
            a_max=norm_clip[1],
            b_min=norm_clip[2],
            b_max=norm_clip[3],
            clip=True,
        ),
        MapLabelsToZeroOutsideRange(
            keys=["label"],
            valid_labels=list(range(14)),
        ),
    ])

    # build the MONAI dataset + loader
    data_dicts = get_data_files(images_dir, labels_dir)
    dataset = Dataset(data=data_dicts, transform=transform)
    dataloader = ThreadDataLoader(
        dataset,
        batch_size=1,
        num_workers=mp.cpu_count(),
    )

    # iterate, transform, and save as .npz
    for batch in tqdm(dataloader, desc="Processing volumes"):
        # extract the single sample
        img_tensor = batch["image"][0]
        lbl_tensor = batch["label"][0]
        base_name = batch["base_name"][0]
        img = img_tensor.numpy().squeeze()
        lbl = lbl_tensor.numpy().squeeze()

        # save as .npz
        np.savez_compressed(
            os.path.join(out_image_dir, f"{base_name}.npz"),
            image=img)
        np.savez_compressed(
            os.path.join(out_label_dir, f"{base_name}.npz"),
            label=lbl)


if __name__ == "__main__":
    images_dir = "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Images"
    labels_dir = "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Labels"
    out_image_dir = "data/preprocessed/val/images_npz"
    out_label_dir = "data/preprocessed/val/labels_npz"
    pixdim = (0.8, 0.8, 1.0)
    norm_clip = (-1024, 1024, -1024, 1024)

    main(
        images_dir, labels_dir,
        out_image_dir, out_label_dir,
        pixdim, norm_clip
    )
