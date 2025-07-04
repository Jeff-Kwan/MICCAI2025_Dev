import torch
from typing import Mapping, Hashable
import monai.transforms as mt
from monai.config import KeysCollection

class QuantizeNormalized(mt.MapTransform):
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
                raise TypeError(f"QuantizeNormalized: expected torch.Tensor for key '{key}', got {type(x)}")
            d[key] = self._quantize(x)
        return d

    @staticmethod
    def _quantize(x: torch.Tensor) -> torch.Tensor:
        # 1) compute channel‐sums
        sums = x.sum(dim=0, keepdim=True)
        sums[sums == 0] = 1.0  # avoid division by zero

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
    

if __name__ == "__main__":
    from time import time
    # instantiate the dictionary-based transform
    quantize = QuantizeNormalized(keys="image")

    # create a dummy input tensor with non‐negative floats
    x = torch.randn(14, 256, 256, 256).abs() * 100
    data = {"image": x}

    # apply the transform
    start_time = time()
    out = quantize(data)
    end_time = time()
    print(f"Quantization took {end_time - start_time:.5f} seconds.")
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