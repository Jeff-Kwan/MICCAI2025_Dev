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
    transform = mt.LoadImaged(["image"], image_only=False),
    dataset = Dataset(data=image_files, transform=transform)
    dataloader = ThreadDataLoader(dataset, batch_size=1, num_workers=120)
    shapes = []
    spacings = []

    dataloader = tqdm(dataloader)
    for data in dataloader:
        img = data["image"]
        # print(f"shape = {tuple(img.shape)}, pixdim = {data['image_meta_dict']['pixdim']}")
        shapes.append(img.shape)
        spacings.append(data['image_meta_dict']['pixdim'][0].numpy()[1:4])

    # Mean, Max, Min
    shapes = np.array(shapes)
    print(f"Shape: Mean: {np.mean(shapes, axis=0)}, Median: {np.median(shapes, axis=0)}, Max: {np.max(shapes, axis=0)}, Min: {np.min(shapes, axis=0)}")
    
    # Shape corresponding to max, median, min numels
    numels = np.prod(shapes, axis=1)
    print(f"Max numel: {np.max(numels)} at shape {shapes[np.argmax(numels)]}")
    print(f"Median numel: {np.median(numels)} at shape {shapes[np.argmin(np.abs(numels - np.median(numels)))]}")
    print(f"Min numel: {np.min(numels)} at shape {shapes[np.argmin(numels)]}")

    # Spacings
    spacings = np.array(spacings)
    print(f"Spacing: Mean: {np.mean(spacings, axis=0)}, Median: {np.median(spacings, axis=0)}, Max: {np.max(spacings, axis=0)}, Min: {np.min(spacings, axis=0)}")


if __name__ == "__main__":
    images_dir = "data/FLARE-Task2-LaptopSeg/validation/Validation-Hidden-Images"
    print_volume_shapes(images_dir)
