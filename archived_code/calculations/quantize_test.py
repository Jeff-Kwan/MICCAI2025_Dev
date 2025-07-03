import torch

def quantize_tensor_dim0(x: torch.Tensor) -> torch.Tensor:
    """
    Quantizes a float tensor along dim=0 so that each slice sums to 255 (uint8),
    while preserving the original proportions as closely as possible.

    Args:
        x: Tensor of shape (C, …) with non-negative floats.
    Returns:
        A uint8 tensor of the same shape, summing to 255 along dim=0.
    """
    sums = x.sum(dim=0, keepdim=True)
    # Avoid division by zero
    sums = torch.where(sums == 0, torch.tensor(1.0, device=sums.device), sums)

    # Scale to 255
    scaled = x * 255.0 / sums

    # Take integer floor
    floors = scaled.floor()
    residuals = scaled - floors

    # Compute number of counts still needed to reach 255 in each slice
    deficits = (255 - floors.sum(dim=0, keepdim=True))

    # Rank residuals in each slice (descending)
    rank = residuals.argsort(dim=0, descending=True).argsort(dim=0)

    # Add 1 to the channels with the highest residuals until sum is 255
    add_one = (rank < deficits).to(torch.uint8)
    result = floors + add_one

    return result.to(torch.uint8)



# Test
x = torch.randn(14, 256, 256, 256).abs() * 100
y = quantize_tensor_dim0(x)
assert y.shape == x.shape, "Output shape mismatch"
assert y.dtype == torch.uint8, "Output dtype should be uint8"
# Check that each slice sums to 255
assert (y.sum(dim=0) == 255).all(), "Each slice should sum to 255"
print("Quantization successful, each slice sums to 255.")

dequant_y = y.float() / 255
assert torch.allclose(dequant_y.sum(dim=0), torch.tensor(1.0)), "Each dequantized slice should sum to 1"
print("Dequantization successful, each slice sums to 1.0")