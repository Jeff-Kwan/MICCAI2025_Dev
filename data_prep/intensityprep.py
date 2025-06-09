import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def get_file_paths(dir, ext=".nii.gz"):
    return [os.path.join(dir, f)
            for f in sorted(os.listdir(dir)) if f.endswith(ext)]

def compute_global_min_max(paths):
    mn, mx = np.inf, -np.inf
    for p in tqdm(paths, desc="Scan min/max"):
        data = nib.load(p).get_fdata(dtype=np.float32)
        mn = min(mn, data.min())
        mx = max(mx, data.max())
    return mn, mx

def compute_histogram(paths, mn, mx, bins=10000):
    hist = np.zeros(bins, dtype=np.int64)
    edges = np.linspace(mn, mx, bins + 1)
    for p in tqdm(paths, desc="Accumulate hist"):
        data = nib.load(p).get_fdata(dtype=np.float32)
        h, _ = np.histogram(data.ravel(), bins=edges)
        hist += h
    return hist, edges

def extract_percentiles(hist, edges, percentiles=[0.5, 99.5]):
    cum = np.cumsum(hist)
    total = cum[-1]
    results = []
    for pct in percentiles:
        threshold = total * pct / 100.0
        bin_idx = np.searchsorted(cum, threshold)
        # linear interpolation:
        prev_cum = cum[bin_idx-1] if bin_idx > 0 else 0
        h = hist[bin_idx]
        frac = (threshold - prev_cum) / (h or 1)
        val = edges[bin_idx] + frac * (edges[bin_idx+1] - edges[bin_idx])
        results.append(val)
    return results

def compute_clipped_stats_from_hist(hist, edges, p_low, p_high):
    # bin centers:
    centers = (edges[:-1] + edges[1:]) * 0.5
    # winsorize them:
    clipped = np.clip(centers, p_low, p_high)
    total = hist.sum()
    # dot‐products to get sum and sum‐of‐squares:
    sum_       = np.dot(hist, clipped)
    sum_sq     = np.dot(hist, clipped * clipped)
    mean       = sum_ / total
    var        = sum_sq / total - mean*mean
    std        = np.sqrt(var)
    return mean, std

if __name__ == "__main__":
    image_paths = get_file_paths("data/preprocessed/train_gt/images")
    mn, mx = compute_global_min_max(image_paths)
    hist, edges = compute_histogram(image_paths, mn, mx, bins=1000000)

    p_low, p_high = extract_percentiles(hist, edges)
    print(f"0.5th percentile = {p_low:.4f}")
    print(f"99.5th percentile = {p_high:.4f}")

    mean_clip, std_clip = compute_clipped_stats_from_hist(hist, edges, p_low, p_high)
    print(f"Mean of clipped intensities = {mean_clip:.4f}")
    print(f"Std  of clipped intensities = {std_clip:.4f}")
