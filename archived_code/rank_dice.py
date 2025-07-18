import torch
from monai.metrics import DiceMetric
import monai.transforms as mt
import os
import glob
import numpy as np
from tqdm import tqdm

val_pred_dir = "archived_code/plots/labels/val_outputs"
val_labels_dir = "archived_code/plots/labels/Validation-Public-Labels"

pred_files = glob.glob(os.path.join(val_pred_dir, "*.nii.gz"))
val_files = glob.glob(os.path.join(val_labels_dir, "*.nii.gz"))

pred_map = {os.path.basename(f): f for f in pred_files}
val_map = {os.path.basename(f): f for f in val_files}

common_names = set(pred_map.keys()) & set(val_map.keys())

results = []
dice_metric = DiceMetric(include_background=False, ignore_empty=False)
loader = mt.Compose([
    mt.LoadImaged(keys=["vol1", "vol2"], ensure_channel_first=True),
    mt.CropForegroundd(keys=["vol1", "vol2"], source_key="vol2"),
    mt.KeepLargestConnectedComponentd(keys=["vol1"]),
    mt.AsDiscreted(keys=["vol1", "vol2"], to_onehot=14),
    mt.EnsureTyped(keys=["vol1", "vol2"], dtype=[torch.long, torch.long], track_meta=False),
])

for name in tqdm(common_names, desc="Calculating Dice"):
    dice_metric.reset()
    data = loader({
        "vol1": pred_map[name],
        "vol2": val_map[name]
    })
    pred = data["vol1"].unsqueeze(0)
    label = data["vol2"].unsqueeze(0)
    results.append((name, dice_metric(pred, label).cpu().numpy()))

results.sort(key=lambda x: x[1].mean())

for name, dice in results:
    print(f"{name}: {np.round(dice, 4)}")

print(f"\nMean Dice: {sum(d[1].mean() for d in results) / len(results):.4f}")