import monai.transforms as mt
import os
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import torch

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.RemoveSmall import RemoveSmallObjectsPerClassd

val_pred_dir = "archived_code/plots/labels/au4_output"
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
    # mt.RemoveSmallObjectsd(keys=["vol1"], min_size=200),
    # mt.KeepLargestConnectedComponentd(keys=["vol1"], independent=True, num_components=2),
    # RemoveSmallObjectsPerClassd(keys=["vol1"]),
    mt.EnsureTyped(keys=["vol1", "vol2"], dtype=np.int64, track_meta=True),
])
ignore_empty = False


num_classes = 14
def compute_stats(name):
    data = loader({
        "vol1": pred_map[name],
        "vol2": val_map[name]
    })
    pred = data["vol1"].cpu().numpy()
    label = data["vol2"].cpu().numpy()

    # 1) load and flatten
    data  = loader({"vol1": pred_map[name], "vol2": val_map[name]})
    pred  = data["vol1"].ravel().long()   # shape (N,)
    label = data["vol2"].ravel().long()   # shape (N,)

    # 2) build confusion matrix: row=true class, col=pred class
    idx  = label * num_classes + pred
    conf = torch.bincount(idx,
                         minlength=num_classes**2
                        ).view(num_classes, num_classes).float()

    # 3) extract the counts for foreground classes 1..num_classes-1
    #    gt_counts[i-1] = total pixels where label==i
    gt_counts    = conf[1:, :].sum(dim=1)

    #    fp_counts[i-1] = #predicted i when ground-truth was 0
    fp_counts    = conf[0, 1:]

    #    fn_counts[i-1] = #predicted 0 when ground-truth was i
    fn_counts    = conf[1:, 0]

    #    wrong_counts[i-1] = #ground-truth i predicted as some j≠i, j>0
    total_pos    = conf[1:, 1:].sum(dim=1)     # all pred>0 for each gt i
    true_pos     = conf.diag()[1:]             # diag element is correct preds
    wrong_counts = total_pos - true_pos

    # 4) compute rates with “if gt_count==0 then (count>0)?1:0”
    denom = gt_counts
    fp_rate    = torch.where(denom>0,
                             fp_counts/denom,
                             (fp_counts>0).float())
    fn_rate    = torch.where(denom>0,
                             fn_counts/denom,
                             (fn_counts>0).float())
    wrong_rate = torch.where(denom>0,
                             wrong_counts/denom,
                             (wrong_counts>0).float())

    return name, fp_rate, fn_rate, wrong_rate

if __name__ == "__main__":
    with Pool(25) as pool:
        results = list(tqdm(pool.imap(compute_stats, common_names), total=len(common_names), desc="Calculating Stats"))

    print("\nPer class mistakes:")
    class_fp = np.mean(np.array([x[1] for x in results]), axis=0)
    class_fn = np.mean(np.array([x[2] for x in results]), axis=0)
    class_wrong = np.mean(np.array([x[3] for x in results]), axis=0)
    print(f"False Positives: \n{np.round(class_fp * 100)}")
    print(f"False Negatives: \n{np.round(class_fn * 100)}")
    print(f"Wrong Class: \n{np.round(class_wrong * 100)}")