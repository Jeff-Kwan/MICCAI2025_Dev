import os
from pathlib import Path
import monai.transforms as mt
from torch.nn import functional as F
import torch
from tqdm import tqdm
from multiprocessing import Pool

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


# Will be initialized in each worker process
def init_worker(weights, soft_labels_dir, extension):
    global load_transform, saver, w1, w2
    w1, w2 = weights

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
    # strip off the channel dim and make sure long for one_hot
    lbl1 = data["lbl1"].squeeze(0)
    lbl2 = data["lbl2"].squeeze(0)
    # one-hot → [H,W,(D),14], permute → [14,H,W,(D)], uint8
    oh1 = F.one_hot(lbl1, num_classes=14).permute(3,0,1,2).to(torch.uint8)
    oh2 = F.one_hot(lbl2, num_classes=14).permute(3,0,1,2).to(torch.uint8)
    # integer mixing
    soft = oh1.mul(w1) + oh2.mul(w2)
    data["soft"] = soft
    saver(data)


if __name__ == "__main__":
    workers = 128
    aladdin5 = "data/nifti/train_pseudo/aladdin5"
    blackbean = "data/nifti/train_pseudo/blackbean"
    soft_labels = "data/nifti/train_pseudo/soft_labels"
    extension = ".nii.gz"
    W1, W2 = 0.6, 0.4
    W1 = int((W1 * 255 + 0.5))
    W2 = 255 - W1
    os.makedirs(soft_labels, exist_ok=True)

    files = get_data_files(aladdin5, blackbean, extension)
    # spawn worker pool
    with Pool(
        processes=workers,
        initializer=init_worker,
        initargs=((W1, W2), soft_labels, extension)
    ) as pool:
        # imap_unordered to allow progress bar
        for _ in tqdm(pool.imap_unordered(process_item, files), total=len(files), desc="Soft-Labelling MP"):
            pass