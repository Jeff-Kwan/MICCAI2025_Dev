import os
from pathlib import Path
import monai.transforms as mt
from monai.networks.utils import one_hot
import torch
from torch.nn.functional import normalize
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import torch


def get_data_files(dir1, dir2, extension=".nii.gz"):
    dir1 = Path(dir1)
    dir2 = Path(dir2)

    names1 = sorted([e.name for e in os.scandir(dir1) if e.is_file() and e.name.endswith(extension)])
    names2 = sorted([e.name for e in os.scandir(dir2) if e.is_file() and e.name.endswith(extension)])
    missing = [n for n in names1 if n not in names2]
    if missing:
        raise ValueError(f"Missing labels: {missing}")
    return [{"lbl1": str(dir1 / n), "lbl2": str(dir2 / n), "base_name": n.removesuffix(extension)} for n in names1]


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


# Will be initialized in each worker process
def init_worker(weights, soft_labels_dir, extension):
    global load_transform, saver, w0, w1
    w0, w1 = weights

    load_transform = mt.Compose([
        mt.LoadImaged(["lbl1", "lbl2"], ensure_channel_first=True),
        mt.EnsureTyped(["lbl1", "lbl2"], dtype=torch.long, track_meta=True),
    ])
    saver = mt.SaveImaged(
        keys=["soft"],
        meta_keys=["lbl1_meta_dict"],
        output_dir=soft_labels_dir,
        output_postfix="",
        output_ext=extension,
        separate_folder=False,
        output_dtype=torch.float32,
        print_log=False
    )
    # Ensure output directory exists
    Path(soft_labels_dir).mkdir(parents=True, exist_ok=True)


def process_item(item):
    data = load_transform(item)
    lbl1 = one_hot(data["lbl1"].unsqueeze(0), num_classes=14).float().squeeze(0)
    lbl2 = one_hot(data["lbl2"].unsqueeze(0), num_classes=14).float().squeeze(0)

    # compute soft labels and normalize
    data["soft"] = quantize_tensor_dim0(normalize(w0 * lbl1 + w1 * lbl2, p=1, dim=0))
    saver(data)


if __name__ == "__main__":
    workers = 160
    aladdin5 = "data/nifti/train_pseudo/aladdin5"
    blackbean = "data/nifti/train_pseudo/blackbean"
    soft_labels = "data/nifti/train_pseudo/soft_labels"
    extension = ".nii.gz"
    weights = [0.6, 0.4]
    os.makedirs(soft_labels, exist_ok=True)

    files = get_data_files(aladdin5, blackbean, extension)
    # spawn worker pool
    with Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(weights, soft_labels, extension)
    ) as pool:
        # imap_unordered to allow progress bar
        for _ in tqdm(pool.imap_unordered(process_item, files), total=len(files), desc="Soft-Labelling MP"):
            pass