import os
from pathlib import Path
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader
import numpy as np
from tqdm import tqdm

def get_data_files(images_dir, extension=".nii.gz"):
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir!r}")

    image_files = [
        {"image": str(images_dir / entry.name)}
        for entry in os.scandir(images_dir)
        if entry.is_file() and entry.name.endswith(extension)
    ]
    if not image_files:
        raise RuntimeError(f"No '{extension}' files found in {images_dir!r}")
    return image_files

def print_volume_shapes(images_dir):
    image_files = get_data_files(images_dir)
    transform = mt.Compose([
        mt.LoadImaged(["image"], image_only=False, ensure_channel_first=True),
        mt.AsDiscreted("image", argmax=True),
        mt.CopyItemsd(["image"], 1, ["image2"]),
        
        # True Foreground
        mt.CropForegroundd("image", "image"),

        # Centercrop Foreground
        mt.CenterSpatialCropd(["image2"], roi_size=(256, 256, -1)),
        mt.CropForegroundd("image2", "image2")])
    
    dataset = Dataset(data=image_files, transform=transform)
    dataloader = ThreadDataLoader(dataset, batch_size=1, num_workers=120)
    shapes_fg = []
    shapes_pfg = []
    
    dataloader = tqdm(dataloader)

    discrepancy = 0
    for data in dataloader:
        true_fg = data["image"].squeeze()
        processed_fg = data["image2"].squeeze()
        if not np.all(true_fg.shape == processed_fg.shape):
            print(f"Found 1 discrepancy: {true_fg.shape} != {processed_fg.shape}")
            discrepancy += 1

        shapes_fg.append(true_fg.shape)
        shapes_pfg.append(processed_fg.shape)

    # Mean, Max, Min
    shapes_fg = np.array(shapes_fg)
    shapes_pfg = np.array(shapes_pfg)
    print(f"True Foreground Shapes: Mean: {np.mean(shapes_fg, axis=0)}, Median: {np.median(shapes_fg, axis=0)}, Max: {np.max(shapes_fg, axis=0)}, Min: {np.min(shapes_fg, axis=0)}")
    print(f"Processed Foreground Shapes: Mean: {np.mean(shapes_pfg, axis=0)}, Median: {np.median(shapes_pfg, axis=0)}, Max: {np.max(shapes_pfg, axis=0)}, Min: {np.min(shapes_pfg, axis=0)}")
    print(f"Discrepancies found: {discrepancy}")

if __name__ == "__main__":
    images_dir = "data/small/train_pseudo/pseudo1x"
    print_volume_shapes(images_dir)

'''
256, 256, -1
No discrepancies for GT or public val

For pseudo 1x, 2 discrepancies
torch.Size([240, 166, 112]) != torch.Size([238, 166, 112])
torch.Size([190, 179, 135]) != torch.Size([190, 177, 135])
'''