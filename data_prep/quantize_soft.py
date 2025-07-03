import os
from pathlib import Path
from tqdm import tqdm
import torch
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader

from typing import Sequence, Hashable, Union, Mapping
from monai.config import KeysCollection


class QuantizeTensorDim0d(mt.MapTransform):
    """
    Dictionary-based MONAI transform to quantize a float tensor along dim=0 so that each slice sums to 255 (uint8),
    preserving original proportions as closely as possible.
    Invert with: mt.NormalizeIntensityd(keys="label", subtrahend=0.0, divisor=255.0)

    Args:
        keys: Key or list of keys in the input dictionary whose values are torch.Tensors to be quantized.
    """

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
        sums = x.sum(dim=0, keepdim=True)
        sums = torch.where(sums == 0, torch.tensor(1.0, device=sums.device), sums)
        scaled = x * 255.0 / sums
        floors = scaled.floor()
        residuals = scaled - floors
        deficits = 255 - floors.sum(dim=0, keepdim=True)
        rank = residuals.argsort(dim=0, descending=True).argsort(dim=0)
        add_one = (rank < deficits).to(torch.uint8)
        result = floors + add_one
        return result.to(torch.uint8)


def process_dataset(in_dir, out_dir):
    # create output dirs
    os.makedirs(out_dir, exist_ok=True)

    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["label"], ensure_channel_first=True),
            mt.EnsureTyped(
                keys=["label"],
                dtype=torch.float32,
                track_meta=True),
            QuantizeTensorDim0d(keys=["label"]),
            mt.SaveImaged(
                keys=["label"],
                output_dir=out_dir,
                output_postfix="",
                output_ext=".nii.gz",
                separate_folder=False,
                output_dtype=torch.uint8,
                print_log=False),
            mt.DeleteItemsd(keys=["label"])
        ]
    )

    # build the MONAI dataset
    dir = Path(in_dir)
    if not dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir!r}")
    names = sorted(
        entry.name
        for entry in os.scandir(dir)
        if entry.is_file() and entry.name.endswith(".nii.gz"))
    dataset = Dataset(data=[{"label": str(dir / name)} for name in names], transform=transform)
    dataloader = ThreadDataLoader(
        dataset,
        batch_size=1,
        num_workers=32,
        prefetch_factor=16)

    # iterate, transform, and save
    for batch in tqdm(dataloader, desc=f"Processing GT to Soft"):
        pass
    return 

if __name__ == "__main__":
    labels = [
        ("data/nifti/train_gt/softlabel",
        "data/nifti/train_gt/softquant"),
        ("data/nifti/train_pseudo/softlabel",
        "data/nifti/train_pseudo/softquant"),
    ]
    for label in labels:
        process_dataset(*label)
        
