import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # Fragmentation
import json
import torch
import monai.transforms as mt
from monai.inferers import sliding_window_inference
from pathlib import Path
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

# --- your model imports ---
from utils.quantize import QuantizeNormalized
from model.AttnUNet3 import AttnUNet3


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


# --- CPU-side post-processing function ---
def cpu_post(data, inference_config):
    prep_tf = mt.Compose([
        mt.LoadImaged(keys=["label"], ensure_channel_first=True),
        mt.EnsureTyped(keys=["label"], dtype=torch.float32),
        mt.NormalizeIntensityd(
            keys=["label"],
            subtrahend=0.0,
            divisor=255.0)
    ])
    post_tf = mt.Compose([
        mt.DeleteItemsd(keys=["pred"]),
        QuantizeNormalized(keys="label"),
        mt.SaveImaged(
            keys=["label"],
            output_dir=inference_config["output_dir"],
            output_postfix="",
            output_ext=".nii.gz",
            separate_folder=False,
            output_dtype=torch.uint8,
            print_log=False),
        mt.DeleteItemsd(keys=["label"])
    ])

    # Prepare
    data = prep_tf(data)
    alpha = inference_config["ema_alpha"]
    class_weights = torch.tensor(inference_config["class_weights"]).view(1, -1, 1, 1, 1)

    # EMA
    data["label"] = alpha * data["label"] + (1-alpha) * data["pred"] * class_weights

    # Quantize and save
    data = post_tf(data)
    return 

@torch.inference_mode()
def run_and_save(
    chunk, inference_config, model, device,
    gpu_id, n_cpu_workers
):
    # Pre-build loader
    loader = mt.LoadImaged(["img"], ensure_channel_first=True)
    deleter = mt.DeleteItemsd(["img"])

    # Inference + dispatch to CPU pool
    max_prefetch = 2
    in_flight = set()
    with ProcessPoolExecutor(max_workers=n_cpu_workers) as executor:
        for pair in tqdm(chunk, desc=f"GPU {gpu_id}", unit="img"):
            try:
                # CPU → GPU prep
                data = loader(pair)
                img = data["img"].to(device).unsqueeze(0)

                # GPU inference
                if inference_config["autocast"]:
                    with torch.autocast("cuda", torch.bfloat16):
                        data["pred"] = torch.softmax(   # softmax logits
                            sliding_window_inference(
                            img,
                            roi_size=inference_config["shape"],
                            sw_batch_size=inference_config.get("sw_batch_size", 1),
                            predictor=model,
                            overlap=inference_config.get("sw_overlap", 0.25),
                            mode="gaussian",
                        ), dim=1).cpu().squeeze(0)
                else:
                    data["pred"] = sliding_window_inference(
                        img,
                        roi_size=inference_config["shape"],
                        sw_batch_size=inference_config.get("sw_batch_size", 1),
                        predictor=model,
                        overlap=inference_config.get("sw_overlap", 0.25),
                        mode="gaussian",
                    ).cpu().squeeze(0)

            except Exception as e:
                print(f"[ERROR] GPU {gpu_id} failed on {pair['img']}: {e}")

            # Done with image
            data = deleter(data)

            # 3) submit to CPU pool
            fut = executor.submit(cpu_post, data, inference_config)
            in_flight.add(fut)

            # 4) if we've queued >= max_prefetch, wait for at least one to finish
            if len(in_flight) >= max_prefetch:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()

        # drain remaining futures
        for f in as_completed(in_flight):
            f.result()


def worker(
    gpu_id, chunks, inference_config,
    model_class, model_config, model_path,
    n_cpu_workers
):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Build & load model
    model = model_class(model_config)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    # Run inference + CPU post
    run_and_save(
        chunk=chunks[gpu_id],
        inference_config=inference_config,
        model=model,
        device=device,
        gpu_id=gpu_id,
        n_cpu_workers=n_cpu_workers,
        )

if __name__ == "__main__":
    # --- configuration ---
    model_class     = AttnUNet3 
    model_config    = json.load(open("configs/labellers/AttnUNet3/model.json", "r"))
    model_path      = "output/2025-07-04/06-44-AttnUNet3/model.pth"

    inference_config = {
        "images_dir": "data/nifti/train_gt/images",
        "labels_dir": "data/nifti/train_gt/softquant",
        "output_dir": "data/nifti/train_gt/softiterated",
        "ema_alpha": 0.5,  # EMA factor keep original label
        "class_weights": [1.0] * 14,
        "shape": [224, 224, 112],
        "sw_batch_size": 2,
        "sw_overlap": 0.5,
        "autocast": True,
    }

    # Prepare data & split
    all_pairs = get_image_label_pairs(inference_config["images_dir"],
                                      inference_config["labels_dir"])
    ngpus     = torch.cuda.device_count()
    np.random.shuffle(all_pairs)
    chunks    = np.array_split(all_pairs, ngpus)

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
                model_class,
                model_config,
                model_path,
                cpus_per_gpu,
            ),
            nprocs=ngpus,
            join=True,
            daemon=False,  # ensure workers are not daemonic
            # No timeout argument is set, so workers can run indefinitely
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main process. Terminating children...")
        mp.get_context('spawn')._shutdown()

    print("Soft pseudo labels updated successfully.")