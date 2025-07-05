import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # Fragmentation
import torch
import json
import subprocess
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
    iterations = 10
    architecture = "ConvSeg"
    model_params = "configs/labellers/ConvSeg/model.json"
    train_params = "configs/labellers/ConvSeg/pseudo_train.json"
    infer_params = "configs/labellers/ConvSeg/pseudo_update.json"
    model_path = "output/Labeller/Base-ConvSeg/model.pth"


    for iter in range(iterations):
        output_dir = f"output/Iterative/Iter_{iter+1}"

        subprocess.run([
            "python", "pseudo_train.py",
            "--architecture", architecture,
            "--output_dir", output_dir,
            "--model_path", model_path,
            "--model_params", model_params,
            "--train_params", train_params,
        ])

        # After training
        model_path = output_dir + "/model.pth"
        # Update inference confidence with dice score
        metrics = json.load(open(output_dir + "/metrics.json", "r"))
        class_dice = metrics["val_metrics"]["class_dice"][-1]
        infer_config = json.load(open(infer_params, "r"))
        infer_config["class_weights"] = class_dice
        with open(infer_params, "w") as f:
            json.dump(infer_config, f, indent=4)


        subprocess.run([
            "python", "pseudo_update.py",
            "--config", infer_params,
            "--model_path", model_path
        ])
