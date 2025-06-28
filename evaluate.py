import multiprocessing as mp
from pathlib import Path

import numpy as np
import monai.transforms as mt
import monai.metrics as mm
from monai.networks.utils import one_hot

def match_pred_label(pred_dir, label_dir, extension=".nii.gz"):
    pred_dir = Path(pred_dir)
    label_dir = Path(label_dir)
    matched = []
    
    # Scan predictions and check for corresponding label
    for pred_path in pred_dir.glob(f"*{extension}"):
        if not pred_path.is_file():
            continue
        stem = pred_path.name.removesuffix(extension)
        label_path = label_dir / f"{stem}{extension}"
        if label_path.is_file():
            matched.append({
                "pred": str(pred_path),
                "label": str(label_path)
            })
    
    return matched

# functional wrappers, so no state is accumulated across calls
def process_pair(pair):
    """
    Loads one pred/label, computes dice & surface dice, returns arrays.
    """
    loader = mt.LoadImaged(keys=["pred", "label"], ensure_channel_first=True)
    data = loader(pair)
    # add batch dim and one-hot encode
    pred = one_hot(data["pred"].unsqueeze(0), num_classes=14)
    label = one_hot(data["label"].unsqueeze(0), num_classes=14)
    # compute per-class dice (exclude background) and surface-dice
    dice_vals = mm.compute_dice(
        y_pred=pred, y=label, include_background=False, ignore_empty=False
    ).numpy()
    surf_vals = mm.compute_surface_dice(
        y_pred=pred, y=label,
        include_background=False,
        class_thresholds=[1]*13
    ).numpy()
    return dice_vals, surf_vals

if __name__ == "__main__":
    # gather your list of dicts just like before
    matched_files = match_pred_label("./docker/outputs", "./docker/labels")
    
    # choose number of workers (e.g. number of cores)
    n_workers = 50
    with mp.Pool(n_workers) as pool:
        # results is a list of (dice_array, surf_array) tuples
        results = pool.map(process_pair, matched_files)
    
    # unpack & stack
    dice_list, surf_list = zip(*results)
    dice_array = np.stack(dice_list, axis=0)      # shape: (N_samples, N_classes-1)
    surf_array = np.stack(surf_list, axis=0)      # same shape
    
    # per-class means
    class_dices = np.nanmean(dice_array, axis=0)  # mean across samples
    class_surfs = np.nanmean(surf_array, axis=0)  # mean across samples

    # overall means
    mean_dice = class_dices.mean()
    mean_surf = class_surfs.mean()

    print("Per-class Dices:", class_dices)
    print("Per-class Surface Dices:", class_surfs)
    print("Mean Dice:", mean_dice, "|||", "Mean Surface Dice:", mean_surf)

    w = (1.01 - class_dices) / (1.01 - class_dices).mean()
    print("Class weights:", [round(float(x), 3) for x in w])
