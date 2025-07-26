from monai.metrics import DiceMetric
import monai.transforms as mt
import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.RemoveSmall import RemoveSmallObjectsPerClassd

val_pred_dir = "archived_code/plots/labels/au5_post_output"
val_labels_dir = "archived_code/plots/labels/Validation-Public-Labels"

pred_files = glob.glob(os.path.join(val_pred_dir, "*.nii.gz"))
val_files = glob.glob(os.path.join(val_labels_dir, "*.nii.gz"))

pred_map = {os.path.basename(f): f for f in pred_files}
val_map = {os.path.basename(f): f for f in val_files}

common_names = list(set(pred_map.keys()) & set(val_map.keys()))

from monai.data import MetaTensor
from skimage.morphology import remove_small_objects
class RemoveSmallObjectsPerClassd(mt.Transform):
    def __init__(self, keys, labels, min_sizes, connectivity=1):
        self.keys = keys
        self.labels = labels
        self.min_sizes = min_sizes
        self.conn = connectivity

    def __call__(self, data):
        for key in self.keys:
            img = data[key].cpu().numpy()
            for lbl, ms in zip(self.labels, self.min_sizes):
                mask = (img == lbl)
                if mask.any():
                    cleaned_mask = remove_small_objects(mask, min_size=ms, connectivity=self.conn)
                    img[mask & (~cleaned_mask)] = 0
            if isinstance(data[key], MetaTensor):
                data[key] = MetaTensor(img, meta=data[key].meta)
            elif isinstance(data[key], torch.Tensor):
                data[key] = torch.tensor(img, dtype=data[key].dtype, device=data[key].device)
            else:
                data[key] = img
        return data
    
loader = mt.Compose([
    mt.LoadImaged(keys=["vol1", "vol2"], ensure_channel_first=True),
    # mt.CropForegroundd(keys=["vol1", "vol2"], source_key="vol2", margin=64, allow_smaller=True),
    # mt.KeepLargestConnectedComponentd(keys=["vol1"], independent=True, num_components=2),
    # mt.RemoveSmallObjectsd(keys=["vol1"], min_size=256),
    # RemoveSmallObjectsPerClassd(keys=["vol1"]),
    mt.AsDiscreted(keys=["vol1", "vol2"], to_onehot=14),
])
ignore_empty = False

def compute_dice(name):
    dice_metric = DiceMetric(include_background=False, ignore_empty=ignore_empty, reduction='none')
    dice_metric.reset()
    data = loader({
        "vol1": pred_map[name],
        "vol2": val_map[name]
    })
    pred = data["vol1"].unsqueeze(0)
    label = data["vol2"].unsqueeze(0)
    dice = dice_metric(pred, label).cpu().numpy()
    return (name, dice)

if __name__ == "__main__":
    with Pool(20) as pool:
        results = list(tqdm(pool.imap(compute_dice, common_names), total=len(common_names), desc="Calculating Dice"))
    mean_dice = sum(np.nanmean(d[1]) for d in results) / len(results)
    results.sort(key=lambda x: x[1].mean())
    results = [[x[0].replace(".nii.gz", "")] + np.round(x[1].squeeze(), 3).tolist() for x in results]

    df = pd.DataFrame(
        results,
        columns=["Name"] + ["Liver", "R-Kidney", "Spleen", "Pancreas", 
                                "Aorta", "IVC", "RAG", "LAG",
                                "Gallbladder", "Esophagus", "Stomach", "Duodenum", "L-Kidney"]
    )
    print(df.to_string(index=False, col_space=8, justify="center"))
    if ignore_empty:
        suffix = "_ignore_empty"
    else:
        suffix = ""
    df.to_csv(f"archived_code/attnunet5_yespost.csv", index=False)

    print(f"\nMean Dice: {mean_dice:.4f}")