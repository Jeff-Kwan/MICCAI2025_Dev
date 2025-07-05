import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # Fragmentation
import torch
import subprocess
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
    iterations = 10
    architecture = "ConvSeg"
    model_params = "configs\labellers\ConvSeg\model.json"
    train_params = "configs\labellers\ConvSeg\pseudo_train.json"
    infer_params = "configs\labellers\ConvSeg\pseudo_update.json"


    for iter in range(iterations):
        output_dir = f"output/Iterative/Iter_{iter}"

        subprocess.run([
            "python", "pseudo_train.py",
            "--architecture", architecture,
            "--output_dir", output_dir,
            "--model_params", model_params,
            "--train_params", train_params,
        ])

        model_path = output_dir + "/model.pth"
        subprocess.run([
            "python", "pseudo_update.py",
            "--config", infer_params,
            "--model_path", model_path
        ])
