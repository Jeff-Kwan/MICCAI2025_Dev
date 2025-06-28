import json
import torch
import monai.transforms as mt
from monai.data import Dataset
from monai.inferers import sliding_window_inference
import monai.metrics as mm
from monai.networks.utils import one_hot
from pathlib import Path
import os
from tqdm import tqdm
from time import time
import multiprocessing as mp
import numpy as np

# My Model
from model.AttnUNet2 import AttnUNet as AttnUNet2
from model.AttnUNet import AttnUNet
from model.ViTSeg import ViTSeg
from model.ConvSeg import ConvSeg


def get_image_files(images_dir, extension=".nii.gz"):
    images_dir = Path(images_dir)
    image_dicts = [
        {"img": str(entry.path)}
        for entry in os.scandir(images_dir)
        if entry.is_file() and entry.name.endswith(extension)
    ]
    return image_dicts


def get_pre_transforms(pixdim, intensities):
    upper, lower, mean, std = intensities
    spatial = mt.Compose([
        mt.LoadImaged(["img"], image_only=False, ensure_channel_first=True),
        mt.EnsureTyped(["img"], dtype=torch.float32, track_meta=True),
        mt.Orientationd(["img"], axcodes="RAS", lazy=False),
        mt.Spacingd(["img"], pixdim=pixdim, mode=("trilinear"), lazy=False),
    ])
    intensity = mt.Compose([
        mt.ThresholdIntensityd(["img"], above=False, threshold=upper, cval=upper),
        mt.ThresholdIntensityd(["img"], above=True, threshold=lower, cval=lower),
        mt.NormalizeIntensityd(["img"], subtrahend=mean, divisor=std),
    ])
    return spatial, intensity

def get_post_transforms(pre_transforms, output_dir):
    return mt.Compose([
        mt.AsDiscreted(keys="pred", argmax=True),   # No need softmax as argmax directly
        mt.Invertd(
            keys="pred",
            transform=pre_transforms,
            orig_keys="img",  
            meta_keys="pred_meta_dict",
            orig_meta_keys="img_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,   # Nearest Neighbour of Predictions
            to_tensor=True),
        mt.SaveImaged(keys="pred",
            output_dir=output_dir, 
            output_postfix="", 
            output_ext=".nii.gz", 
            resample=False,     # Invert already resamples
            separate_folder=False,
            output_dtype=torch.uint8,
            print_log=False)
        ])


@torch.inference_mode()
def run_inference(in_dir, out_dir, model, inference_config, device):
    model.eval().to(device)

    # Create dataset and dataloader
    spatial_tf, intensity_tf = get_pre_transforms(inference_config["pixdim"], 
                                                  inference_config["intensities"])
    post_tf = get_post_transforms(spatial_tf, out_dir)
    dataset = Dataset(
        data=get_image_files(in_dir), 
        transform=mt.Compose([spatial_tf, intensity_tf]))
    os.makedirs(out_dir, exist_ok=True)

    # Run inference
    times = []
    total_start_time = time()
    for data in tqdm(dataset, desc="Inference"):
        start_time = time()
        data["pred"] = sliding_window_inference(
                    data["img"].to(device, non_blocking=True).unsqueeze(0),
                    roi_size=inference_config['shape'],
                    sw_batch_size=inference_config.get('sw_batch_size', 1),
                    predictor=lambda x: model(x),
                    overlap=inference_config.get('sw_overlap', 0.25),
                    mode="gaussian",
                    buffer_steps=1).cpu().squeeze(0)
        times.append(time() - start_time)
        # Post-processing and saving results
        post_tf(data)

        print(sum(times) / len(times), "seconds per image")
        
    total_time = time() - total_start_time
    print(f"Total inference time: {total_time:.2f} seconds")
    # Print average data processing time and average inference time
    print(f"Average inference time: {sum(times) / len(times):.2f} seconds")
    print(f"Average data loading time: {(total_time - sum(times)) / len(times):.2f} seconds")



def match_pred_label(pred_dir, label_dir, extension=".nii.gz"):
    pred_dir = Path(pred_dir)
    label_dir = Path(label_dir)
    matched = []
    
    # Scan predictions and check for corresponding label
    for pred_path in pred_dir.glob(f"*{extension}"):
        if not pred_path.is_file():
            continue
        stem = pred_path.name.removesuffix(extension)
        label_path = label_dir / f"{stem}{extension}"
        if label_path.is_file():
            matched.append({
                "pred": str(pred_path),
                "label": str(label_path)
            })
    
    return matched

# functional wrappers, so no state is accumulated across calls
def process_pair(pair):
    """
    Loads one pred/label, computes dice & surface dice, returns arrays.
    """
    loader = mt.LoadImaged(keys=["pred", "label"], ensure_channel_first=True)
    data = loader(pair)
    # add batch dim and one-hot encode
    pred = one_hot(data["pred"].unsqueeze(0), num_classes=14)
    label = one_hot(data["label"].unsqueeze(0), num_classes=14)
    # compute per-class dice (exclude background) and surface-dice
    dice_vals = mm.compute_dice(
        y_pred=pred, y=label, include_background=False, ignore_empty=False
    ).numpy()
    surf_vals = mm.compute_surface_dice(
        y_pred=pred, y=label,
        include_background=False,
        class_thresholds=[1]*13
    ).numpy()
    return dice_vals, surf_vals


if __name__ == "__main__":
    model = AttnUNet(json.load(open(r'./configs/labellers/AttnUNet/model.json', 'r')))
    # model = ViTSeg(json.load(open(r'./configs/labellers/ViTSeg/model.json', 'r')))
    # model = ConvSeg(json.load(open(r'./configs/labellers/ConvSeg/model.json', 'r')))
    model.load_state_dict(torch.load(r"output/2025-06-28/15-24-AttnUNet/model.pth", weights_only=True))
    inference_config = {
        "pixdim": [0.8, 0.8, 2.5],
        "intensities": [295.0, -974.0, 95.958, 139.964],
        "shape": [256, 256, 128],
        "sw_batch_size": 2,
        "sw_overlap": 0.5,
    }
    device = torch.device("cuda:0")
    in_dir = "./data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Images"
    label_dir = "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Labels"
    out_dir = "./data/inference"
    run_inference(in_dir, out_dir, model, inference_config, device)
    print("Inference completed. Now evaluating...")

    # gather your list of dicts just like before
    matched_files = match_pred_label(out_dir, label_dir)
    with mp.Pool(50) as pool:
        # results is a list of (dice_array, surf_array) tuples
        results = pool.map(process_pair, matched_files)
    
    # unpack & stack
    dice_list, surf_list = zip(*results)
    dice_array = np.stack(dice_list, axis=0)      # shape: (N_samples, N_classes-1)
    surf_array = np.stack(surf_list, axis=0)      # same shape
    
    # per-class means
    class_dices = np.nanmean(dice_array, axis=0)  # mean across samples
    class_surfs = np.nanmean(surf_array, axis=0)  # mean across samples

    # overall means
    mean_dice = class_dices.mean()
    mean_surf = class_surfs.mean()

    print("Per-class Dices:", class_dices)
    print("Per-class Surface Dices:", class_surfs)
    print("Mean Dice:", mean_dice, "|||", "Mean Surface Dice:", mean_surf)

    w = (1.01 - class_dices) / (1.01 - class_dices).mean()
    print("Class weights:", [round(float(x), 3) for x in w])