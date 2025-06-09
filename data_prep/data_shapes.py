import os
from pathlib import Path
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader
from tqdm import tqdm

def get_data_files(images_dir, extension=".nii.gz"):
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir!r}")

    image_files = sorted(
        str(images_dir / entry.name)
        for entry in os.scandir(images_dir)
        if entry.is_file() and entry.name.endswith(extension)
    )
    if not image_files:
        raise RuntimeError(f"No '{extension}' files found in {images_dir!r}")
    return image_files

def print_volume_shapes(images_dir):
    image_files = get_data_files(images_dir)
    transform = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True),
    ])
    dataset = Dataset(data=image_files, transform=transform)
    dataloader = ThreadDataLoader(dataset, batch_size=1, num_workers=4)
    for idx, data in enumerate(tqdm(dataloader, desc="Volumes")):
        img = data[0]
        print(f"{os.path.basename(image_files[idx])}: shape = {tuple(img.shape)}")

if __name__ == "__main__":
    images_dir = "data/FLARE-Task2-LaptopSeg/train_gt_label/imagesTr"
    print_volume_shapes(images_dir)
