import argparse
import json
import torch
import monai.transforms as mt
from monai.data import Dataset
from monai.inferers import sliding_window_inference
from pathlib import Path
import os

# My Model
from model.AttnUNet2 import AttnUNet


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
        mt.Orientationd(["img"], axcodes="RAS"),
        mt.Spacingd(["img"], pixdim=pixdim, mode=("bilinear")),
        # mt.CropForegroundd(
        #         keys=["img"],
        #         source_key="img",
        #         select_fn=lambda x: (x > lower) & (x < upper),
        #         allow_smaller=True),
    ])
    intensity = mt.Compose([
        mt.ThresholdIntensityd(["img"], above=False, threshold=upper, cval=upper),
        mt.ThresholdIntensityd(["img"], above=True, threshold=lower, cval=lower),
        mt.NormalizeIntensityd(["img"], subtrahend=mean, divisor=std),
    ])
    return spatial, intensity

def get_post_transforms(pre_transforms, output_dir):
    return mt.Compose([
        mt.Invertd(
            keys="pred",
            transform=pre_transforms,
            orig_keys="img",  
            meta_keys="pred_meta_dict",
            orig_meta_keys="img_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,   # Smooth interpolation of logits
            to_tensor=True),
        mt.AsDiscreted(keys="pred", argmax=True),   # No need softmax as argmax directly
        mt.SaveImaged(keys="pred",
            output_dir=output_dir, 
            output_postfix="", 
            output_ext=".nii.gz", 
            resample=False,     # Invert already resamples
            separate_folder=False,
            output_dtype=torch.uint8)
        ])


@torch.inference_mode()
def run_inference(args, inference_config):
    # Load the model
    model = AttnUNet(json.load(open('./model/attn_unet.json', 'r')))
    model.load_state_dict(torch.load(args.model_weight, weights_only=True))
    model.eval().to(args.device)

    # Create dataset and dataloader
    spatial_tf, intensity_tf = get_pre_transforms(inference_config["pixdim"], 
                                                  inference_config["intensities"])
    post_tf = get_post_transforms(spatial_tf, args.output_dir)
    dataset = Dataset(
        data=get_image_files(args.inputs_dir), 
        transform=mt.Compose([spatial_tf, intensity_tf]))
    # dataloader = DataLoader(
    #     dataset, 
    #     batch_size=1, 
    #     pin_memory=True,
    #     num_workers=inference_config["workers"])
    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference
    from tqdm import tqdm
    from time import time
    times = []
    total_start_time = time()
    for data in tqdm(dataset, desc="Inference"):
        start_time = time()
        data["pred"] = sliding_window_inference(
                    data["img"].to(args.device, non_blocking=True).unsqueeze(0),
                    roi_size=inference_config['shape'],
                    sw_batch_size=inference_config.get('sw_batch_size', 1),
                    predictor=lambda x: model(x),
                    overlap=inference_config.get('sw_overlap', 0.25),
                    mode="gaussian").cpu().squeeze(0)
        times.append(time() - start_time)
        # Post-processing and saving results
        # data["pred_meta_dict"] = data["img_meta_dict"]
        post_tf(data)
        print(sum(times) / len(times), "seconds per image")
    total_time = time() - total_start_time
    print(f"Total inference time: {total_time:.2f} seconds")
    # Print average data processing time and average inference time
    print(f"Average inference time: {sum(times) / len(times):.2f} seconds")
    print(f"Average data loading time: {(total_time - sum(times)) / len(times):.2f} seconds")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs_dir', type=str, default=r'./inputs', help='dir of output')
    parser.add_argument('--output_dir', type=str, default=r'./outputs', help='dir of output')
    parser.add_argument('--model_weight', type=str, default=r'./model/model.pth', help='weight')
    parser.add_argument('--device', type=str, default='cuda', help='device to run inference on')
    args = parser.parse_args()

    inference_config = json.load(open('./inference_config.json', 'r'))
    run_inference(args, inference_config)