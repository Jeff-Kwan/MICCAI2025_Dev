import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from flare_dataset import FLARESegDataset_npy  # your original dataset class

# Global dataset variable for each worker
dataset = None

def init_worker(img_dir, label_dir, spacing_dir):
    """
    Initialize global dataset variable in each worker process.
    """
    global dataset
    dataset = FLARESegDataset_npy(img_dir, label_dir, spacing_dir)

def process_case(idx):
    """
    Process a single case: loads image/label/spacing, computes shape, aspect ratios, and per-case label counts.
    """
    global dataset
    sample = dataset[idx]
    img = sample["image"].numpy()   # (H, W, D)
    lbl = sample["label"].numpy()   # (H, W, D)
    spacing = sample["spacing"]
    fname = sample["filename"]

    H, W, D = img.shape
    ar_xy = W / H
    ar_xz = W / D
    ar_yz = H / D
    label_counts_full = np.bincount(lbl.ravel())
    label_counts = np.zeros(14, dtype=np.int64)
    label_counts[:min(14, label_counts_full.shape[0])] = label_counts_full[:14]
    if label_counts_full.shape[0] > 14:
        print(f"Warning: case {fname} contains unexpected label values: {np.unique(lbl)}")
    return (idx, fname, (H, W, D), spacing, (ar_xy, ar_xz, ar_yz), label_counts)

def main(img_dir, label_dir, spacing_dir):
    # For total count up front
    serial_dataset = FLARESegDataset_npy(img_dir, label_dir, spacing_dir)
    n_cases = len(serial_dataset)
    print(f"Found {n_cases} cases in train_gt_label.\n")

    results = []

    n_cpus = max(1, mp.cpu_count() - 1)
    print(f"Using up to {n_cpus} parallel worker(s)...\n")

    # Launch Pool with init_worker
    with mp.Pool(processes=n_cpus, initializer=init_worker, initargs=(img_dir, label_dir, spacing_dir)) as pool:
        for res in tqdm(pool.imap_unordered(process_case, range(n_cases)), total=n_cases, desc="Processing cases"):
            results.append(res)
        pool.close()
        pool.join()

    # Sort results by original index
    results.sort(key=lambda x: x[0])

    all_shapes = []
    all_spacings = []
    global_label_counts = np.zeros(14, dtype=np.int64)

    for res in results:
        idx, fname, shape, spacing, aspect_ratios, label_counts = res
        all_shapes.append(shape)
        all_spacings.append(spacing)
        global_label_counts += label_counts

    # === GLOBAL SUMMARY ===
    print("\n\n=== GLOBAL SUMMARY OVER ALL TRAIN_GT_LABEL CASES ===")
    shapes_np = np.array(all_shapes)
    H_vals = shapes_np[:, 0]
    W_vals = shapes_np[:, 1]
    D_vals = shapes_np[:, 2]

    print("Voxel‐dim (D):  min = {:d}, max = {:d},  mean = {:.1f}".format(
        int(D_vals.min()), int(D_vals.max()), float(D_vals.mean())))
    print("Voxel‐dim (H):  min = {:d}, max = {:d},  mean = {:.1f}".format(
        int(H_vals.min()), int(H_vals.max()), float(H_vals.mean())))
    print("Voxel‐dim (W):  min = {:d}, max = {:d},  mean = {:.1f}".format(
        int(W_vals.min()), int(W_vals.max()), float(W_vals.mean())))

    spac_np = np.array(all_spacings)
    sx, sy, sz = spac_np[:, 0], spac_np[:, 1], spac_np[:, 2]
    print("\nSpacing (x):  min = {:.3f}, max = {:.3f},  mean = {:.3f}  (mm)".format(
        float(sx.min()), float(sx.max()), float(sx.mean())))
    print("Spacing (y):  min = {:.3f}, max = {:.3f},  mean = {:.3f}  (mm)".format(
        float(sy.min()), float(sy.max()), float(sy.mean())))
    print("Spacing (z):  min = {:.3f}, max = {:.3f},  mean = {:.3f}  (mm)".format(
        float(sz.min()), float(sz.max()), float(sz.mean())))

    ar_xy_list = (W_vals / H_vals)
    ar_xz_list = (W_vals / D_vals)
    ar_yz_list = (H_vals / D_vals)
    print("\nAspect ratio (W/H):  min = {:.3f}, max = {:.3f}, mean = {:.3f}".format(
        float(ar_xy_list.min()), float(ar_xy_list.max()), float(ar_xy_list.mean())))
    print("Aspect ratio (W/D):  min = {:.3f}, max = {:.3f}, mean = {:.3f}".format(
        float(ar_xz_list.min()), float(ar_xz_list.max()), float(ar_xz_list.mean())))
    print("Aspect ratio (H/D):  min = {:.3f}, max = {:.3f}, mean = {:.3f}".format(
        float(ar_yz_list.min()), float(ar_yz_list.max()), float(ar_yz_list.mean())))

    # Human-readable labels
    label_map = {
        0: "background",
        1: "liver",
        2: "right-kidney",
        3: "spleen",
        4: "pancreas",
        5: "aorta",
        6: "ivc",
        7: "rag",
        8: "lag",
        9: "gallbladder",
        10: "esophagus",
        11: "stomach",
        12: "duodenum",
        13: "left kidney"
    }

    total_voxels = global_label_counts.sum()
    print(f"\nOverall class balance (total voxels across all {n_cases} GT cases = {total_voxels:,d}):")
    for v in range(len(global_label_counts)):
        cnt = global_label_counts[v]
        lbl_name = label_map.get(int(v), f"Label {int(v)}")
        pct = 100.0 * cnt / total_voxels
        print(f"  • {lbl_name} (value={v}): {cnt:,d} voxels  ({pct:.2f}%)")

if __name__ == "__main__":
    img_dir = "./data/FLARE-Task2-LaptopSeg/data_npy/images"
    label_dir = "./data/FLARE-Task2-LaptopSeg/data_npy/labels"
    spacing_dir = "./data/FLARE-Task2-LaptopSeg/data_npy/spacings"
    main(img_dir, label_dir, spacing_dir)
