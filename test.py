from torch.utils.data import DataLoader

# Suppose you have:
#   ./data/images/   → contains CT volumes as <basename>.npy (converted from .nii.gz)
#   ./data/labels/   → contains segmentation masks as <basename>.npy
images_folder = "./data/images"
labels_folder = "./data/labels"

# (Optionally) define a simple normalization + random crop transform, e.g.:
def example_transform(sample):
    img, lbl = sample["image"], sample["label"]
    # Example: normalize each volume to zero mean / unit std
    img = (img - img.mean()) / (img.std() + 1e-8)
    # No change to label in this simple transform
    return {"image": img, "label": lbl, "metadata": sample["metadata"]}

# Instantiate dataset
dataset = CTSegmentationDataset(
    images_dir=images_folder,
    labels_dir=labels_folder,
    transform=example_transform,  # or None if you do transforms elsewhere
)

# Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,      # adjust to your CPU
    pin_memory=True,    # if training on GPU
)

# Iterate
for batch in loader:
    imgs = batch["image"]       # shape = [B, 1, D, H, W]
    masks = batch["label"]      # shape = [B, 1, D, H, W]
    metas = batch["metadata"]   # list of length B (each element is a dict or None)
    # Now feed `imgs` and `masks` into your 3D segmentation model...
    break
