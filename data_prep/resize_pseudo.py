import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader
from monai.config import KeysCollection
from typing import Hashable, Mapping

class QuantizeTensorDim0d(mt.MapTransform):
    def __init__(self, keys: KeysCollection) -> None:
        super().__init__(keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict:
        # make a shallow copy so we don't modify the original dict
        d = dict(data)
        for key in self.keys:
            x = d[key]
            if not torch.is_tensor(x):
                raise TypeError(f"QuantizeTensorDim0d: expected torch.Tensor for key '{key}', got {type(x)}")
            d[key] = self._quantize(x)
        return d

    @staticmethod
    def _quantize(x: torch.Tensor) -> torch.Tensor:
        # 1) compute channel‐sums and clamp zeros to 1 in-place
        sums = x.sum(dim=0, keepdim=True)
        sums.clamp_(min=1.0)                     # avoid div-by-zero via in-place clamp :contentReference[oaicite:0]{index=0}

        # 2) scale each channel so sum→255
        scaled = x.mul(255.0).div_(sums)

        # 3) get integer floor via a single cast and compute residuals
        floors = scaled.to(torch.uint8)          # float→uint8 is a truncation cast :contentReference[oaicite:1]{index=1}
        residuals = scaled - floors.float()

        # 4) compute how many “ones” to distribute per spatial location
        deficits = (255 - floors.sum(dim=0, keepdim=True))

        # 5) vectorize the “largest‐residual” selection
        C = x.size(0)
        # flatten spatial dims into one axis
        res_flat = residuals.view(C, -1)         # shape [C, N]
        def_flat = deficits.view(-1)            # shape [N]

        # single sort along channel axis
        _, idx_flat = res_flat.sort(dim=0, descending=True)  # one sort call :contentReference[oaicite:2]{index=2}

        # build a mask in sorted order: for each pixel j, top def_flat[j] channels get +1
        dr = torch.arange(C, device=x.device).view(C, 1)
        mask_sorted = dr < def_flat.unsqueeze(0)            # shape [C, N]

        # scatter the mask back to original channel positions
        mask_flat = torch.zeros_like(mask_sorted)
        mask_flat.scatter_(0, idx_flat, mask_sorted)

        # reshape mask to [C, ...] and form final result
        mask = mask_flat.view_as(x).to(torch.uint8)
        return (floors + mask).to(torch.uint8)

def get_data_files(labels_dir, extension=".nii.gz"):
    labels_dir = Path(labels_dir)

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels_dir!r}")

    label_names = sorted(
        entry.name
        for entry in os.scandir(labels_dir)
        if entry.is_file() and entry.name.endswith(extension)
    )
    if not label_names:
        raise RuntimeError(f"No '{extension}' files found in {labels_dir!r}")

    return [
        {"label": str(labels_dir / name), "base_name": name.removesuffix(".nii.gz")}
        for name in label_names
    ]

def process_pseudo(datafiles, output_dir):
    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["label"], image_only=False, ensure_channel_first=True),
            mt.EnsureTyped(keys=["label"], dtype=np.float32, track_meta=True),
            mt.Orientationd(keys=["label"], axcodes="RAS"),
            mt.Spacingd(keys=["label"], pixdim=(1.6, 1.6, 2.5), mode="trilinear"),
            QuantizeTensorDim0d(keys=["label"]),
            mt.SaveImaged(
                keys=["label"],
                output_dir=output_dir,
                output_postfix="",
                output_ext=".nii.gz",
                separate_folder=False,
                output_dtype=torch.uint8,
                print_log=False),
            mt.DeleteItemsd(keys=["label"]),
        ]
    )
    dataloader = ThreadDataLoader(
        Dataset(data=datafiles, transform=transform),
        batch_size=1,
        num_workers=36,
        pin_memory=False,
        persistent_workers=True)
    
    for batch in tqdm(dataloader, desc="Processing pseudo-labels"):
        pass

if __name__ == "__main__":
    datafiles = get_data_files("data/nifti/train_pseudo/pseudo1x")
    output_dir = "data/small/train_pseudo/pseudo1x"

    process_pseudo(datafiles, output_dir)