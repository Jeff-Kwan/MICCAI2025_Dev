import os
from pathlib import Path
from tqdm import tqdm
import torch
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader

def get_data_files(images_dir, labels_dir, extension=".nii.gz"):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir!r}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels_dir!r}")

    image_names = sorted(
        entry.name
        for entry in os.scandir(images_dir)
        if entry.is_file() and entry.name.endswith(extension)
    )
    if not image_names:
        raise RuntimeError(f"No '{extension}' files found in {images_dir!r}")

    label_names = {
        entry.name
        for entry in os.scandir(labels_dir)
        if entry.is_file() and entry.name.endswith(extension)
    }
    if not label_names:
        raise RuntimeError(f"No '{extension}' files found in {labels_dir!r}")

    missing = [name for name in image_names if name not in label_names]
    if missing:
        missing_list = ", ".join(repr(n) for n in missing)
        raise ValueError(f"Missing labels for images: {missing_list}")

    return [
        {"image": str(images_dir / name), "label": str(labels_dir / name),
         "base_name": name.removesuffix(extension)}
        for name in image_names
    ]

def filter_and_delete_zeros(datafiles):
    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["label"]),
            mt.EnsureTyped(keys=["label"], dtype=torch.uint8, track_meta=True),
        ]
    )

    dataset = Dataset(data=datafiles, transform=transform)
    dataloader = ThreadDataLoader(dataset, batch_size=1, num_workers=8)

    for data in tqdm(dataloader, desc="Filtering zero-label images"):
        label = data["label"][0].numpy().squeeze()

        if (label == 0).all():
            label_path = data['label'][0].meta['filename_or_obj']
            base_name = Path(label_path).name
            matching_files = [f for f in datafiles if Path(f['label']).name == base_name]
            if matching_files:
                image_path = matching_files[0]['image']
                print(f"Deleting image and label with all-zero label:\n  Image: {image_path}\n  Label: {label_path}")
                try:
                    os.remove(image_path)
                    os.remove(label_path)
                except Exception as e:
                    print(f"Error deleting files: {e}")

if __name__ == "__main__":
    datafiles = get_data_files(
        "data/preprocessed/val/images",
        "data/preprocessed/val/labels"
    )
    datafiles += get_data_files(
        "data/preprocessed/train_pseudo/images",
        "data/preprocessed/train_pseudo/aladdin5"
    )

    filter_and_delete_zeros(datafiles)
