import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def get_file_paths(dir, ext=".nii.gz"):
    return [os.path.join(dir, f)
            for f in sorted(os.listdir(dir)) if f.endswith(ext)]

# Worker for min/max
def _min_max(path):
    data = nib.load(path).get_fdata(dtype=np.float32)
    return float(data.min()), float(data.max())

def compute_global_min_max(paths, n_procs=24):
    with Pool(n_procs) as p:
        # imap yields results as they come in
        results = list(tqdm(p.imap(_min_max, paths, chunksize=10),
                            total=len(paths), desc="Scan min/max"))
    # unzip mins and maxs
    mins, maxs = zip(*results)
    return min(mins), max(maxs)

# Worker for histogram on a single image
def _histogram_for_path(args):
    path, edges = args
    data = nib.load(path).get_fdata(dtype=np.float32).ravel()
    h, _ = np.histogram(data, bins=edges)
    return h

def compute_histogram(paths, mn, mx, bins=10000, n_procs=24):
    # build common edges once
    edges = np.linspace(mn, mx, bins + 1, dtype=np.float32)
    with Pool(n_procs) as p:
        # prepare args: each worker gets (path, edges)
        args = ((p, edges) for p in paths)
        # map and accumulate
        hist = np.zeros(bins, dtype=np.int64)
        for h in tqdm(p.imap(_histogram_for_path, args, chunksize=5),
                      total=len(paths), desc="Accumulate hist"):
            hist += h
    return hist, edges

def extract_percentiles(hist, edges, percentiles=[0.5, 99.5]):
    cum = np.cumsum(hist)
    total = cum[-1]
    results = []
    for pct in percentiles:
        threshold = total * pct / 100.0
        idx = np.searchsorted(cum, threshold)
        prev_cum = cum[idx-1] if idx > 0 else 0
        h = hist[idx]
        # avoid division by zero
        frac = (threshold - prev_cum) / (h or 1)
        val = edges[idx] + frac * (edges[idx+1] - edges[idx])
        results.append(val)
    return results

def compute_clipped_stats_from_hist(hist, edges, p_low, p_high):
    centers = 0.5 * (edges[:-1] + edges[1:])
    clipped = np.clip(centers, p_low, p_high)
    total = hist.sum()
    sum_   = np.dot(hist, clipped)
    sum_sq = np.dot(hist, clipped * clipped)
    mean   = sum_ / total
    var    = sum_sq / total - mean*mean
    std    = np.sqrt(var)
    return mean, std

if __name__ == "__main__":
    image_paths = (
        get_file_paths("data/preprocessed/train_gt/images") +
        get_file_paths("data/preprocessed/train_pseudo/images")
    )
    mn, mx = compute_global_min_max(image_paths)
    hist, edges = compute_histogram(image_paths, mn, mx, bins=100000)

    p_low, p_high = extract_percentiles(hist, edges)
    print(f"0.5th percentile = {p_low:.4f}")
    print(f"99.5th percentile = {p_high:.4f}")

    mean_clip, std_clip = compute_clipped_stats_from_hist(
        hist, edges, p_low, p_high
    )
    print(f"Mean of clipped intensities = {mean_clip:.4f}")
    print(f"Std  of clipped intensities = {std_clip:.4f}")


'''
On spatially preprocessed data, 100000 bins
0.5th percentile = -3023.9445
99.5th percentile = 497.6330
Mean of clipped intensities = -582.4940
Std  of clipped intensities = 561.3401
'''