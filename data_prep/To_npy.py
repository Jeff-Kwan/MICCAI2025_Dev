from pathlib import Path
import numpy as np
import nibabel as nib
import json
from tqdm import tqdm
import concurrent.futures
import multiprocessing

def process_single_nii(nii_path_str: str):
    """
    Worker function to process a single .nii.gz file:
      - Load image
      - Save image data as .npy
      - Extract all header metadata (including spacing) + affine, save as JSON
      - Delete the original .nii.gz

    Returns:
      None on success, or an error message (str) on failure.
    """
    nii_path = Path(nii_path_str)
    try:
        # Load with nibabel (mmap=True to avoid duplicating memory excessively)
        img = nib.load(str(nii_path), mmap=True)

        # Get the array data as float32
        data_np = img.get_fdata(dtype=np.float32)

        # Determine the basename without ".nii.gz"
        base_name = nii_path.name[:-7]

        # Save the image data as <basename>.npy
        data_save_path = nii_path.with_name(f"{base_name}.npy")
        np.save(str(data_save_path), data_np)

        # Extract header metadata
        header = img.header
        metadata = {}

        # Iterate over all header keys
        for key in header.keys():
            try:
                val = header[key]
                # Convert to numpy array if possible, then to list
                arr = np.array(val)
                metadata[key] = arr.tolist()
            except Exception:
                try:
                    # Fallback: try to convert to list directly
                    metadata[key] = val.tolist()
                except Exception:
                    # Final fallback: convert to string
                    metadata[key] = str(val)

        # Include the affine matrix
        metadata["affine"] = img.affine.tolist()

        # Save all metadata (including spacing) as JSON
        meta_save_path = nii_path.with_name(f"{base_name}_metadata.json")
        with open(meta_save_path, "w") as f:
            json.dump(metadata, f)

        # Remove the original .nii.gz
        nii_path.unlink()
        return None

    except Exception as e:
        return str(e)


def convert_directory_to_npy(dir_path: str, recursive: bool = False):
    """
    Scans `dir_path` for all .nii.gz files (recursively if requested),
    converts each one into:
      - a data‐array .npy
      - a metadata JSON containing header + affine (including spacing)
    and then deletes the original .nii.gz.

    Uses parallel processing across multiple CPU cores.

    Args:
        dir_path (str): Root directory to search for .nii.gz.
        recursive (bool): If True, walk subfolders as well (default False).

    Raises:
        FileNotFoundError: If `dir_path` does not exist or is not a directory.
        RuntimeError: If no `.nii.gz` files are found.
    """
    root = Path(dir_path)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path!r}")

    # Collect all .nii.gz files (either only in dir_path or recursively)
    if recursive:
        nii_paths = sorted(root.rglob("*.nii.gz"))
    else:
        nii_paths = sorted(root.glob("*.nii.gz"))

    if len(nii_paths) == 0:
        raise RuntimeError(f"No .nii.gz files found in {dir_path} (recursive={recursive})")

    # Convert Path objects to strings for pickling in ProcessPoolExecutor
    nii_path_strs = [str(p) for p in nii_paths]

    max_workers = max(1, multiprocessing.cpu_count() - 1)

    # Use a ProcessPoolExecutor to process files in parallel
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_single_nii, p_str): p_str
            for p_str in nii_path_strs
        }

        # Iterate over futures as they complete, with a progress bar
        for future in tqdm(
            concurrent.futures.as_completed(future_to_path),
            total=len(nii_path_strs),
            desc="Converting .nii.gz → .npy",
            ncols=80,
        ):
            origin_path = future_to_path[future]
            err = None
            try:
                err = future.result()
            except Exception as e:
                # An unexpected exception from the worker
                err = str(e)

            if err is not None:
                errors.append((origin_path, err))
                print(f"Warning: failed to process {Path(origin_path).name}: {err}")

    print(f"\nDone! Processed {len(nii_paths)} file(s).")
    if errors:
        print(f"{len(errors)} file(s) encountered errors. See warnings above.")


# =========================
# Example usage:
# =========================
if __name__ == "__main__":
    my_dir = "./data"
    convert_directory_to_npy(
        dir_path=my_dir,
        recursive=True,        # Set True if you want to walk subfolders
    )
