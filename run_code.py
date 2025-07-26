import subprocess

if __name__ == "__main__":
    # subprocess.run(["python3", "ddp_fine_labellers.py"])
    # subprocess.run(["python3", "inference.py"])
    subprocess.run(["python3", "pseudo_update.py", 
        "--config", "configs/labellers/AttnUNet4/pseudo_update.json",
        "--model_path", "output/2025-07-25/16-19-AttnUNet4/model.pth"
    ])