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
      - Extract only the required header metadata (dtype, shape, zooms, affine,
        slice_duration, toffset), each wrapped in its own try/except block
      - Delete the original .nii.gz

    Returns:
      None on success, or an error message (str) on failure.
    """
    nii_path = Path(nii_path_str)
    try:
        # Load with nibabel (mmap=True to avoid duplicating memory excessively)
        img = nib.load(str(nii_path), mmap=True)

        # Get the stored data type and then load the array in that type if uint8,
        # otherwise cast to float32
        data_np = img.get_fdata(dtype=np.float32)

        if img.get_data_dtype() == np.uint8:
            data_np = data_np.astype(np.uint8)

        # Determine the basename without ".nii.gz"
        base_name = nii_path.name[:-7]
        base_name = base_name.replace("_0000.", ".")

        # Save the image data as <basename>.npy
        data_save_path = nii_path.with_name(f"{base_name}.npy")
        np.save(str(data_save_path), data_np)

        # Extract only the required header metadata
        header = img.header
        metadata = {}

        # 1) Data type (e.g. "int16", "float32", etc.)
        try:
            metadata["dtype"] = str(header.get_data_dtype())
        except Exception:
            pass

        # 2) Shape (e.g. [512, 512, 270] for a 3D volume)
        try:
            shape_tuple = header.get_data_shape()
            # Convert to list so JSON can serialize it
            metadata["shape"] = list(shape_tuple)
        except Exception:
            pass

        # 3) Voxel sizes / zooms (e.g. [0.85546875, 0.85546875, 1.0])
        try:
            zooms_tuple = header.get_zooms()
            metadata["zooms"] = [float(z) for z in zooms_tuple]
        except Exception:
            pass

        # 4) Affine transformation matrix (4×4)
        try:
            # img.affine is a numpy array; convert to nested lists
            metadata["affine"] = img.affine.tolist()
        except Exception:
            pass

        # 5) qform sform
        try:
            metadata["qform_code"] = int(header.get("qform_code"))
            metadata["sform_code"] = int(header.get("sform_code"))
        except Exception:
            pass

        # 6) toffset (time offset of the first slice; usually 0.0 for static volumes)
        try:
            toff = header.get("toffset")
            metadata["toffset"] = float(toff) if toff is not None else 0.0
        except Exception:
            pass

        # If you also need scaling parameters (optional):
        try:
            slope = header.get("scl_slope")
            if slope is not None:
                metadata["scl_slope"] = float(slope)
        except Exception:
            pass

        try:
            inter = header.get("scl_inter")
            if inter is not None:
                metadata["scl_inter"] = float(inter)
        except Exception:
            pass

        # Save this minimal metadata as JSON
        meta_save_path = nii_path.with_name(f"{base_name}_metadata.json")
        with open(meta_save_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Remove the original .nii.gz
        nii_path.unlink()

        return None

    except Exception as e:
        # Return the error message so the caller can report which file failed
        return str(e)


def convert_directory_to_npy(dir_path: str, recursive: bool = False):
    """
    Scans `dir_path` for all .nii.gz files (recursively if requested),
    converts each one into:
      - a data‐array .npy (in the original dtype or float32)
      - a minimal metadata JSON containing (dtype, shape, zooms, affine,
        slice_duration, toffset, optionally scl_slope/inter)
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
