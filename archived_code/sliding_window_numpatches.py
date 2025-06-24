# from monai.inferers import SlidingWindowSplitter
# import torch

# image_size = [512, 512, 160]
# roi_size = [256, 256, 128]
# overlap = 0.25

# # Create a dummy image with shape [B, C, D, H, W]
# image = torch.zeros((1, 1, *image_size))

# splitter = SlidingWindowSplitter(patch_size=roi_size, overlap=overlap)
# patches = list(splitter(image))

# print(f"Total number of patches: {len(patches)}")

import math

def compute_num_patches(image_size, roi_size, overlap):
    step = [int(r * (1 - overlap)) for r in roi_size]
    return [math.ceil((i - r) / s) + 1 for i, r, s in zip(image_size, roi_size, step)]

image_size = [512, 512, 160]
roi_size = [224, 224, 112]
overlap = 0.25

num_patches = compute_num_patches(image_size, roi_size, overlap)
total_patches = math.prod(num_patches)

print(f"Number of patches along each dimension: {num_patches}")
print(f"Total number of patches: {total_patches}")

