import subprocess

if __name__ == "__main__":
    subprocess.run(["python3", "ddp_fine_small.py"])
    subprocess.run(["python3", "inference.py"])
    subprocess.run(["python3", "archived_code/rank_dice.py"])
    # subprocess.run(["python3", "pseudo_update2x.py", 
    #     "--config", "configs/labellers/AttnUNet5/pseudo_update2.json",
    #     "--model_path", "output/Labeller/AttnUNet5-Pass2/model.pth"
    # ])