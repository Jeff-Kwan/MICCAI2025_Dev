import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader
from multiprocessing import Pool

def get_data_files(labels_dir, extension = ".nii.gz"):
    labels_dir = Path(labels_dir)
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels_dir!r}")

    label_names = {
        entry.name
        for entry in os.scandir(labels_dir)
        if entry.is_file() and entry.name.endswith(extension)
    }
    if not label_names:
        raise RuntimeError(f"No '{extension}' files found in {labels_dir!r}")

    return [
        {"label": str(labels_dir / name),
         "base_name": name.removesuffix(".nii.gz")}
        for name in label_names
    ]

def process_label_tensor(label, num_classes=14):
    label = label["label"]
    label = label.numpy().squeeze() / 255
    p_mass = label.sum(axis=(1, 2, 3)).squeeze()
    freq = p_mass / p_mass.sum()
    # count = np.bincount(label.numpy().squeeze().ravel(), minlength=num_classes)
    # freq = count / count.sum()
    return freq

if __name__ == "__main__":
    datafiles = get_data_files("data/small/train_pseudo/pseudo1x")

    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["label"], ensure_channel_first=True),
            mt.EnsureTyped(keys=["label"], dtype=np.float32),
            # mt.EnsureTyped(keys=["label"], dtype=np.uint8),
        ]
    )

    dataset = Dataset(data=datafiles, transform=transform)
    loader = ThreadDataLoader(dataset, num_workers=48, batch_size=1, shuffle=False)

    freq_results = []
    with Pool(processes=32) as pool:
        with tqdm(total=len(datafiles), desc="Processing labels") as pbar:
            for label in loader:
                async_result = pool.apply_async(process_label_tensor, (label,))
                freqs = async_result.get()  # Immediately process next batch in parallel
                freq_results.append(freqs)
                pbar.update(1)

    label_frequencies = np.mean(np.array(freq_results), axis=0)
    label_frequencies = torch.tensor(label_frequencies).squeeze()
    label_frequencies *= 100
    print("Label frequencies (%):", [f"{f:.4f}" for f in label_frequencies.tolist()])
