import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # Fragmentation
import torch
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')

def plot_results(metrics, output_dir):
        class_names = ["Liver", "Right kidney", "Spleen", "Pancreas", 
                        "Aorta", "Inferior Vena Cava", "Right Adrenal Gland", 
                        "Gallbladder", "Esophagus", "Stomach", "Duodenum", "Left kidney"]
        train_losses = metrics["train_losses"]
        val_losses = metrics["val_losses"]
        val_metrics = metrics["val_metrics"]
        epochs = range(1, len(train_losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss curve
        ax1.plot(epochs, train_losses, label='Train')
        ax1.plot(epochs, val_losses, label='Val')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.legend(); ax1.set_title('Loss')

        # Dice curve
        ax2.plot(epochs, val_metrics['dice'], label='Val Dice')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice')
        ax2.legend(); ax2.set_title('Validation Dice')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'))
        plt.close(fig)

        # Plot class dice
        plt.figure(figsize=(12, 6))
        class_dice = np.array(val_metrics["class_dice"]).transpose()[1:].tolist()
        for name, dice in zip(class_names, class_dice):
            plt.plot(dice, label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.title("Dice Score for Each Organ over Training")
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_dice.png'))
        plt.close()


if __name__ == "__main__":
    iterations = 10
    architecture = "ConvSeg"
    model_params = "configs/labellers/ConvSeg/model.json"
    train_params = "configs/labellers/ConvSeg/pseudo_train.json"
    infer_params = "configs/labellers/ConvSeg/pseudo_update.json"
    model_path = "output/Labeller/Base-ConvSeg/model.pth"
    output_dir_base = "output/Iterative"

    plot_metrics = {"train_losses": [], "val_losses": [], "val_metrics": {'dice': [], 'class_dice': []}}
    for iter in range(iterations):
        output_dir = f"{output_dir_base}/Iter_{iter+1}"

        # Create new labels
        subprocess.run([
            "python", "pseudo_update.py",
            "--config", infer_params,
            "--model_path", model_path
        ])

        # Fine tune process
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
        

        # plot overall graphs
        plot_metrics["train_losses"].append(metrics["train_losses"])
        plot_metrics["val_losses"].append(metrics["val_losses"])
        plot_metrics["val_metrics"]["dice"].append(metrics["val_metrics"]["dice"])
        plot_metrics["val_metrics"]["class_dice"].append(metrics["val_metrics"]["class_dice"])
        plot_results(plot_metrics, output_dir_base)
            
