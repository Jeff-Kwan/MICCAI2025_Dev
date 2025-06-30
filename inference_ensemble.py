import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # Fragmentation
import json
import torch
import torch.nn.functional as F
import monai.transforms as mt
from monai.inferers import sliding_window_inference
from monai.transforms import LoadImaged
from monai.metrics import compute_dice, compute_surface_dice
from monai.networks.utils import one_hot
from pathlib import Path
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

# --- your model imports ---
from model.AttnUNet import AttnUNet
from model.ViTSeg import ViTSeg
from model.ConvSeg import ConvSeg

def get_image_label_pairs(images_dir, labels_dir, extension=".nii.gz"):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    pairs = []
    for img_path in images_dir.glob(f"*{extension}"):
        lbl_path = labels_dir / img_path.name
        if lbl_path.is_file():
            pairs.append({"img": str(img_path), "label": str(lbl_path)})
        else:
            print(f"[WARN] no label found for {img_path.name}")
    print(f"[INFO] found {len(pairs)} image/label pairs")
    return pairs

def get_pre_transforms(pixdim, intensities):
    upper, lower, mean, std = intensities
    spatial = mt.Compose([
        mt.LoadImaged(["img"], image_only=False, ensure_channel_first=True),
        mt.EnsureTyped(["img"], dtype=[torch.float32], track_meta=True),
        mt.Orientationd(["img"], axcodes="RAS", lazy=False),
        mt.Spacingd(["img"], pixdim=pixdim, mode="trilinear", lazy=False),
    ])
    intensity = mt.Compose([
        mt.ThresholdIntensityd(["img"], above=False, threshold=upper, cval=upper),
        mt.ThresholdIntensityd(["img"], above=True, threshold=lower, cval=lower),
        mt.NormalizeIntensityd(["img"], subtrahend=mean, divisor=std),
    ])
    return spatial, intensity

def get_post_transforms(pre_transforms):
    return mt.Compose([
        mt.GaussianSmoothd(keys=["AttnUNet", "ConvSeg", "ViTSeg"], sigma=0.5),
        mt.MeanEnsembled(
            keys=["AttnUNet", "ConvSeg", "ViTSeg"],
            output_key="pred",
            weights=torch.tensor([[1.0, 1.002, 1.003, 1.008, 1.039, 1.013, 1.017, 1.039, 
                                   1.048, 0.995, 1.056, 1.015, 1.057, 0.996], 
                                   [1.0, 1.007, 1.004, 1.009, 1.025, 1.009, 1.023, 1.097, 
                                    1.111, 1.309, 1.036, 1.002, 1.044, 0.996], 
                                    [1.0, 0.991, 0.993, 0.983, 0.935, 0.977, 0.959, 0.864, 
                                     0.841, 0.696, 0.909, 0.983, 0.899, 1.008]]).view(3, 14, 1, 1, 1)
            ),
        mt.DeleteItemsd(keys=["AttnUNet", "ConvSeg", "ViTSeg"]),
        mt.AsDiscreted(keys="pred", argmax=True),
        mt.Invertd(
            keys="pred",
            transform=pre_transforms,
            orig_keys="img",
            meta_keys="pred_meta_dict",
            orig_meta_keys="img_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        ),
    ])

# --- CPU-side post-processing function (must be top-level for pickling) ---
def cpu_post(pair, data, inference_config, num_classes):
    # Reconstruct inverter
    spatial_tf, _ = get_pre_transforms(
        inference_config["pixdim"], inference_config["intensities"]
    )
    inverter = get_post_transforms(spatial_tf)

    # Prepare data for inversion
    pred_inv = inverter(data)["pred"].unsqueeze(0)

    # Load and one-hot label
    lbl_data = LoadImaged(["label"], ensure_channel_first=True)({"label": pair["label"]})
    label = lbl_data["label"].unsqueeze(0)

    pred_oh = one_hot(pred_inv, num_classes=num_classes)
    lbl_oh  = one_hot(label, num_classes=num_classes)

    dice_vals = compute_dice(
        y_pred=pred_oh, y=lbl_oh,
        include_background=False, ignore_empty=False
    ).numpy()
    # surf_vals = compute_surface_dice(
    #     y_pred=pred_oh, y=lbl_oh,
    #     include_background=False,
    #     class_thresholds=[1]*(num_classes-1)
    # ).numpy()

    return dice_vals, None#, surf_vals

@torch.inference_mode()
def run_and_score(
    chunk, inference_config, models, device,
    shared_metrics, gpu_id, autocast, n_cpu_workers, num_classes
):
    # Pre-build loader
    spatial_tf, intensity_tf = get_pre_transforms(
        inference_config["pixdim"], inference_config["intensities"]
    )
    loader = mt.Compose([spatial_tf, intensity_tf])

    # Inference + dispatch to CPU pool
    max_prefetch = 2
    in_flight = set()
    dtype = torch.bfloat16 if autocast else torch.float32

    with ProcessPoolExecutor(max_workers=n_cpu_workers) as executor:
        for pair in tqdm(chunk, desc=f"GPU {gpu_id}", unit="img"):
            try:
                # CPU → GPU prep
                data = loader(pair)
                img = data["img"].to(device).unsqueeze(0)

                # GPU inference
                for key in models.keys():
                    with torch.autocast("cuda", dtype):
                        data[key] = F.log_softmax(sliding_window_inference(
                            img,
                            roi_size=models[key]["shape"],
                            sw_batch_size=inference_config.get("sw_batch_size", 1),
                            predictor=models[key]["model"],
                            overlap=inference_config.get("sw_overlap", 0.25),
                            mode="gaussian",
                        ).cpu().squeeze(0), dim=0)

            except Exception as e:
                print(f"[ERROR] GPU {gpu_id} failed on {pair['img']}: {e}")

        # 3) submit to CPU pool
            fut = executor.submit(cpu_post, pair, data, inference_config, num_classes)
            in_flight.add(fut)

            # 4) if we've queued >= max_prefetch, wait for at least one to finish
            if len(in_flight) >= max_prefetch:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for f in done:
                    dice_vals, surf_vals = f.result()
                    shared_metrics.append((dice_vals, surf_vals))

        # drain remaining futures
        for f in as_completed(in_flight):
            dice_vals, surf_vals = f.result()
            shared_metrics.append((dice_vals, surf_vals))


def worker(
    gpu_id, chunks, inference_config, model_init,
    shared_metrics, autocast, n_cpu_workers, num_classes
):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Build & load model
    models = {key: {} for key in model_init.keys()}
    for key in model_init.keys():
        models[key]["model"] = model_init[key]["class"](model_init[key]["config"])
        state = torch.load(model_init[key]["path"], map_location=device, weights_only=True)
        models[key]["model"].load_state_dict(state)
        models[key]["model"].to(device).eval()
        models[key]["shape"] = model_init[key]["shape"]

    # Run inference + CPU post
    run_and_score(
        chunk=chunks[gpu_id],
        inference_config=inference_config,
        models=models,
        device=device,
        shared_metrics=shared_metrics,
        gpu_id=gpu_id,
        autocast=autocast,
        n_cpu_workers=n_cpu_workers,
        num_classes=num_classes,
    )

if __name__ == "__main__":
    # --- configuration ---
    autocast        = True
    num_classes     = 14
    model_init = {
        "AttnUNet": {
            "class": AttnUNet,
            "config": json.load(open("configs/labellers/AttnUNet/model.json", "r")),
            "path": "output/Labeller/Base-AttnUNet/model.pth",
            "shape": [256, 256, 112]},
        "ConvSeg": {
            "class": ConvSeg,
            "config": json.load(open("configs/labellers/ConvSeg/model.json", "r")),
            "path": "output/Labeller/Base-ConvSeg/model.pth",
            "shape": [224, 224, 128]},
        "ViTSeg": {
            "class": ViTSeg,
            "config": json.load(open("configs/labellers/ViTSeg/model.json", "r")),
            "path": "output/Labeller/Base-ViTSeg/model.pth",
            "shape": [192, 192, 96]}
    }

    inference_config = {
        "pixdim": [0.8, 0.8, 2.5],
        "intensities": [295.0, -974.0, 95.958, 139.964],
        "sw_batch_size": 2,
        "sw_overlap": 0.5,
    }

    images_dir      = "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Images"
    labels_dir      = "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Labels"

    # Prepare data & split
    all_pairs = get_image_label_pairs(images_dir, labels_dir)
    ngpus     = torch.cuda.device_count()
    np.random.shuffle(all_pairs)
    chunks    = np.array_split(all_pairs, ngpus)

    # Shared list for metrics
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    metrics = manager.list()

    # Decide how many CPU workers per GPU (e.g. total_cpus // ngpus)
    total_cpus = 48
    cpus_per_gpu = max(1, total_cpus // ngpus)

    # Spawn one process per GPU
    try:
        mp.spawn(
            fn=worker,
            args=(
                chunks,
                inference_config,
                model_init,
                metrics,
                autocast,
                cpus_per_gpu,
                num_classes
            ),
            nprocs=ngpus,
            join=True,
            daemon=False,  # ensure workers are not daemonic
            # No timeout argument is set, so workers can run indefinitely
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main process. Terminating children...")
        mp.get_context('spawn')._shutdown()

    # Aggregate & print
    dice_list, surf_list = zip(*list(metrics))
    dice_array = np.stack(dice_list, axis=0)
    # surf_array = np.stack(surf_list, axis=0)

    class_dices = np.nanmean(dice_array, axis=0).squeeze()
    # class_surfs = np.nanmean(surf_array, axis=0).squeeze()
    mean_dice   = class_dices.mean()
    # mean_surf   = class_surfs.mean()

    print("Per-class Dices:", class_dices)
    # print("Per-class Surface Dices:", class_surfs)
    print("Mean Dice:", mean_dice)
    # print("Mean Surface Dice:", mean_surf)

    # weights = list((1.01 - class_dices) / (1.01 - class_dices).mean())
    # print("Class weights:", [round(w, 3) for w in weights])
