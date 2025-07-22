from monai.metrics import DiceMetric
import monai.transforms as mt
import os
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd

val_pred_dir = "archived_code/plots/labels/attnunet3_output"
val_labels_dir = "archived_code/plots/labels/Validation-Public-Labels"

pred_files = glob.glob(os.path.join(val_pred_dir, "*.nii.gz"))
val_files = glob.glob(os.path.join(val_labels_dir, "*.nii.gz"))

pred_map = {os.path.basename(f): f for f in pred_files}
val_map = {os.path.basename(f): f for f in val_files}

common_names = list(set(pred_map.keys()) & set(val_map.keys()))

loader = mt.Compose([
    mt.LoadImaged(keys=["vol1", "vol2"], ensure_channel_first=True),
    # mt.CropForegroundd(keys=["vol1", "vol2"], source_key="vol2", margin=64, allow_smaller=True),
    # mt.KeepLargestConnectedComponentd(keys=["vol1"]),
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
    with Pool(2) as pool:
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
    df.to_csv(f"archived_code/attnunet3{suffix}.csv", index=False)

    print(f"\nMean Dice: {mean_dice:.4f}")