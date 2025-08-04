from monai.metrics import DiceMetric, compute_surface_dice
import monai.transforms as mt
import os
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

val_pred_dir = "archived_code/plots/labels/au6_official"
val_labels_dir = "archived_code/plots/labels/Validation-Public-Labels"

pred_files = glob.glob(os.path.join(val_pred_dir, "*.nii.gz"))
val_files = glob.glob(os.path.join(val_labels_dir, "*.nii.gz"))

pred_map = {os.path.basename(f): f for f in pred_files}
val_map = {os.path.basename(f): f for f in val_files}

common_names = list(set(pred_map.keys()) & set(val_map.keys()))


loader = mt.Compose([
    mt.LoadImaged(keys=["vol1", "vol2"], ensure_channel_first=True, image_only=False),
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

def compute_NSD(name):
    data = loader({
        "vol1": pred_map[name],
        "vol2": val_map[name]
    })
    pred = data["vol1"].unsqueeze(0)
    label = data["vol2"].unsqueeze(0)
    nsd = compute_surface_dice(
        y_pred=pred, y=label,
        include_background=False,
        class_thresholds=[1]*13,
        # spacing=data["vol1"].meta["pixdim"][1:4].tolist()
    ).cpu().numpy()
    return (name, nsd)

if __name__ == "__main__":
    result_name = "au6_official"

    # Dice
    with Pool(50) as pool:
        results = list(tqdm(pool.imap(compute_dice, common_names), total=len(common_names), desc="Calculating Dice"))
    mean_dice = sum(np.nanmean(d[1]) for d in results) / len(results)
    results.sort(key=lambda x: x[1].mean())
    results = [[x[0].replace(".nii.gz", "")] + x[1].squeeze().tolist() for x in results]

    df = pd.DataFrame(
        results,
        columns=["Name"] + ["Liver", "R-Kidney", "Spleen", "Pancreas", 
                                "Aorta", "IVC", "RAG", "LAG",
                                "Gallbladder", "Esophagus", "Stomach", "Duodenum", "L-Kidney"]
    )
    df.to_csv(f"archived_code/{result_name}_dice.csv", index=False)
    print(f"\nMean Dice: {mean_dice:.4f}")

    # NSD
    with Pool(50) as pool:
        results = list(tqdm(pool.imap(compute_NSD, common_names), total=len(common_names), desc="Calculating NSD"))
    mean_nsd = sum(np.nanmean(d[1]) for d in results) / len(results)
    results.sort(key=lambda x: x[1].mean())
    results = [[x[0].replace(".nii.gz", "")] + x[1].squeeze().tolist() for x in results]

    df = pd.DataFrame(
        results,
        columns=["Name"] + ["Liver", "R-Kidney", "Spleen", "Pancreas", 
                                "Aorta", "IVC", "RAG", "LAG",
                                "Gallbladder", "Esophagus", "Stomach", "Duodenum", "L-Kidney"]
    )
    df.to_csv(f"archived_code/{result_name}_nsd.csv", index=False)
    print(f"\nMean NSD: {mean_nsd:.4f}")