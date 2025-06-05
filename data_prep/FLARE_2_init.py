import os
from huggingface_hub import snapshot_download
import subprocess

'''
Make sure we have 7z first:
sudo apt install p7zip-full
'''

# Download the FLARE-2 dataset for Task 2: Laptop Segmentation
local_dir = "./data/FLARE-Task2-LaptopSeg"
os.makedirs(local_dir, exist_ok=True)
snapshot_download(
    repo_id="FLARE-MedFM/FLARE-Task2-LaptopSeg",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)


# Unzip the specific .7z file using 7z command-line tool
file1 = "./data/FLARE-Task2-LaptopSeg/train_pseudo_label/pseudo_label_aladdin5_flare22.7z"
file2 = "./data/FLARE-Task2-LaptopSeg/train_pseudo_label/pseudo_label_blackbean_flare22.zip"
output_dir = "./data/FLARE-Task2-LaptopSeg/train_pseudo_label"
subprocess.run([
    "7z", "x", file1, f"-o{output_dir}"
], check=True)
os.remove(file1)
subprocess.run([
    "7z", "x", file2, f"-o{output_dir}"
], check=True)
os.remove(file2)