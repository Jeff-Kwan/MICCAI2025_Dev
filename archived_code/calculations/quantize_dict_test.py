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
        # sums along channel dim
        sums = x.sum(dim=0, keepdim=True)
        # avoid zero-division
        sums = torch.where(sums == 0, torch.tensor(1.0, device=sums.device), sums)
        # scale so that channel-sums become 255
        scaled = x * 255.0 / sums
        # floor and compute residuals
        floors = scaled.floor()
        residuals = scaled - floors
        # how many counts remain to hit 255 per-location
        deficits = 255 - floors.sum(dim=0, keepdim=True)
        # rank residuals descending (0 is largest)
        rank = residuals.argsort(dim=0, descending=True).argsort(dim=0)
        # add 1 where rank < deficit
        add_one = (rank < deficits).to(torch.uint8)
        result = floors + add_one
        return result.to(torch.uint8)


# instantiate the dictionary-based transform
quantize = QuantizeTensorDim0d(keys="image")

# create a dummy input tensor with non‐negative floats
x = torch.randn(14, 256, 256, 256).abs() * 100
data = {"image": x}

# apply the transform
out = quantize(data)
y = out["image"]

# basic checks
assert y.shape == x.shape, f"Output shape mismatch: got {y.shape}, expected {x.shape}"
assert y.dtype == torch.uint8, f"Output dtype should be uint8, got {y.dtype}"

# Check that each spatial location sums to 255 along dim=0
sums = y.sum(dim=0)
assert (sums == 255).all(), "Each slice should sum to 255"
print("Quantization successful: each slice sums to 255.")

# dequantize and verify sums to 1.0
dequant = mt.NormalizeIntensityd(
    keys="image",
    subtrahend=0.0, 
    divisor=255.0)
dequant_y = dequant(out)["image"]
dequant_sums = dequant_y.sum(dim=0)
# allow a tiny epsilon for floating‐point error
eps = 1e-6
assert torch.allclose(dequant_sums, torch.ones_like(dequant_sums), atol=eps), \
    "Each dequantized slice should sum to 1.0"
print("Dequantization successful: each slice sums to 1.0.")