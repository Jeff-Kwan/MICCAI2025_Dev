from huggingface_hub import snapshot_download

# Download the FLARE-2 dataset for Task 2: Laptop Segmentation
local_dir = "./data/FLARE-Task2-LaptopSeg"
snapshot_download(
    repo_id="FLARE-MedFM/FLARE-Task2-LaptopSeg",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)