import multiprocessing as mp
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import monai.transforms as mt
from monai.inferers import sliding_window_inference
import numpy as np
from monai.networks.utils import one_hot
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import os

from model.AttnUNet import AttnUNet
from model.ViTSeg import ViTSeg
from model.ConvSeg import ConvSeg

def get_pseudo_data(images_dir, aladdin, blackbean, extension=".nii.gz"):
    images_dir = Path(images_dir)
    aladdin = Path(aladdin)
    blackbean = Path(blackbean)
    data = []
    for img_path in images_dir.glob(f"*{extension}"):
        aladdin_path = aladdin / img_path.name
        blackbean_path = blackbean / img_path.name
        data.append({
            "img": str(img_path),
            "aladdin": str(aladdin_path),
            "blackbean": str(blackbean_path)
        })
    print(f"[INFO] found {len(data)} image/label data")
    return data

def get_pre_transforms(pixdim, intensities):
    upper, lower, mean, std = intensities
    spatial = mt.Compose([
        mt.LoadImaged(["img", "aladdin", "blackbean"], image_only=False, ensure_channel_first=True),
        mt.EnsureTyped(["img", "aladdin", "blackbean"], dtype=[torch.float32, torch.long, torch.long], track_meta=True),
        # mt.Orientationd(["img", "aladdin", "blackbean"], axcodes="RAS", lazy=False),
        # mt.Spacingd(["img", "aladdin", "blackbean"], pixdim=pixdim, mode="trilinear", lazy=False),
    ])
    # intensity = mt.Compose([
    #     mt.ThresholdIntensityd(["img"], above=False, threshold=upper, cval=upper),
    #     mt.ThresholdIntensityd(["img"], above=True, threshold=lower, cval=lower),
    #     mt.NormalizeIntensityd(["img"], subtrahend=mean, divisor=std),
    # ])
    return spatial, mt.Identityd(["img"])

# def get_post_transforms(pre_transforms):
#     return mt.Compose([
#         mt.AsDiscreted(keys="pred", argmax=True),
#         mt.Invertd(
#             keys="pred",
#             transform=pre_transforms,
#             orig_keys="img",
#             meta_keys="pred_meta_dict",
#             orig_meta_keys="img_meta_dict",
#             meta_key_postfix="meta_dict",
#             nearest_interp=True,
#             to_tensor=True,
#         ),
#     ])

# --- CPU-side post-processing function (must be top-level for pickling) ---
def cpu_post(data, num_classes, output_dir):
    # Reconstruct inverter
    # spatial_tf, _ = get_pre_transforms(
    #     inference_config["pixdim"], inference_config["intensities"]
    # )
    # inverter = get_post_transforms(spatial_tf)

    saver = mt.SaveImaged(
        keys=["pseudo"],
        output_dir=output_dir,
        output_postfix="",
        resample=False,
        dtype=torch.float32,
        separate_folder=False,
    )

    # Load and one-hot label
    aladdin = data["aladdin"].unsqueeze(0)
    blackbean = data["blackbean"].unsqueeze(0)

    AttnUNet_pred = data["AttnUNet"]
    ConvSeg_pred = data["ConvSeg"]
    ViTSeg_pred = data["ViTSeg"]
    aladdin_oh  = one_hot(aladdin, num_classes=num_classes)
    blackbean_oh = one_hot(blackbean, num_classes=num_classes)

    weights = []
    soft_lbl = weights[0] * AttnUNet_pred + \
                weights[1] * ConvSeg_pred + \
                weights[2] * ViTSeg_pred
    soft_lbl = 0.75 * soft_lbl + 0.15 * aladdin_oh + 0.1 * blackbean_oh
    data["pseudo"] = soft_lbl
    saver(data)



    

@torch.inference_mode()
def run_inference(
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
                        data[key] = F.softmax(
                        sliding_window_inference(
                            img,
                            roi_size=models[key]["shape"],
                            sw_batch_size=inference_config.get("sw_batch_size", 1),
                            predictor=models[key]["model"],
                            overlap=inference_config.get("sw_overlap", 0.25),
                            mode="gaussian",
                        ), dim=1).cpu()

            except Exception as e:
                print(f"[ERROR] GPU {gpu_id} failed on {pair['img']}: {e}")

        # 3) submit to CPU pool
            fut = executor.submit(cpu_post, data, num_classes, output_dir=inference_config["output_dir"])
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
    run_inference(
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
        "output_dir": "data/nifti/train_pseudo/pseudo",
        "pixdim": [0.8, 0.8, 2.5],
        "intensities": [295.0, -974.0, 95.958, 139.964],
        "sw_batch_size": 2,
        "sw_overlap": 0.5,
    }
    os.makedirs(inference_config["output_dir"], exist_ok=True)
    images_dir      = "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Images"
    labels_dir      = "data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Labels"

    # Prepare data & split
    all_pairs = get_pseudo_data(images_dir, labels_dir)
    ngpus     = torch.cuda.device_count()
    np.random.shuffle(all_pairs)
    chunks    = np.array_split(all_pairs, ngpus)

    # Shared list for metrics
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    metrics = manager.list()

    # Decide how many CPU workers per GPU (e.g. total_cpus // ngpus)
    total_cpus = 64
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
