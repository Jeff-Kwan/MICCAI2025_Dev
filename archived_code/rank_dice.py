import torch
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
import monai.transforms as mt
import os
import glob
import nibabel as nib
from tqdm import tqdm

val_outputs_dir = "archived_code/plots/labels/val_outputs"
public_labels_dir = "archived_code/plots/labels/Validation-Public-Labels"

val_files = glob.glob(os.path.join(val_outputs_dir, "*.nii.gz"))
public_files = glob.glob(os.path.join(public_labels_dir, "*.nii.gz"))

val_map = {os.path.basename(f): f for f in val_files}
public_map = {os.path.basename(f): f for f in public_files}

common_names = set(val_map.keys()) & set(public_map.keys())

results = []
dice_metric = DiceMetric(include_background=False, ignore_empty=False)
loader = mt.Compose([
    mt.LoadImaged(keys=["vol1", "vol2"], ensure_channel_first=True),
    mt.CropForegroundd(keys=["vol1", "vol2"], source_key="vol2"),
    mt.KeepLargestConnectedComponentd(keys=["vol1"]),
    mt.EnsureTyped(keys=["vol1", "vol2"], dtype=[torch.long, torch.long], track_meta=False)
])

for name in tqdm(common_names, desc="Calculating Dice"):
    dice_metric.reset()
    data = loader({
        "vol1": val_map[name],
        "vol2": public_map[name]
    })
    val_tensor = data["vol1"].unsqueeze(0)
    pub_tensor = data["vol2"].unsqueeze(0)

    val_tensor = one_hot(val_tensor, num_classes=14)
    pub_tensor = one_hot(pub_tensor, num_classes=14)

    dice = dice_metric(val_tensor, pub_tensor).mean().item()
    results.append((name, dice))

results.sort(key=lambda x: x[1])

for name, dice in results:
    print(f"{name}: {dice:.4f}")