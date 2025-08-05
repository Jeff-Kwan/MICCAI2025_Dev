import argparse
import json
import torch
import monai.transforms as mt
from monai.data import Dataset
from monai.inferers import sliding_window_inference
from pathlib import Path
import os
from tqdm import tqdm
from monai.transforms import Transform
from monai.data import MetaTensor
from skimage.morphology import remove_small_objects


# My Model
from model.AttnUNet6 import AttnUNet6


def get_image_files(images_dir, extension=".nii.gz"):
    images_dir = Path(images_dir)
    image_dicts = [
        {"img": str(entry.path)}
        for entry in os.scandir(images_dir)
        if entry.is_file() and entry.name.endswith(extension)
    ]
    return image_dicts


class RemoveSmallObjectsPerClassd(Transform):
    def __init__(self, keys, 
            labels=list(range(1, 14)),
            min_sizes=[1e4, 1e3, 1e3, 1e3, 1e3, 1e3, 50, 100, 300, 100, 1000, 500, 500],
            connectivity=3):
        self.keys = keys
        self.labels = labels
        self.min_sizes = min_sizes
        self.conn = connectivity

    def __call__(self, data):
        for key in self.keys:
            img = data[key].cpu().numpy()
            for lbl, ms in zip(self.labels, self.min_sizes):
                mask = (img == lbl)
                if mask.any():
                    cleaned_mask = remove_small_objects(mask, min_size=ms, connectivity=self.conn)
                    img[mask & (~cleaned_mask)] = 0
        
            if isinstance(data[key], MetaTensor):
                data[key] = MetaTensor(img, meta=data[key].meta)
            elif isinstance(data[key], torch.Tensor):
                data[key] = torch.tensor(img, dtype=data[key].dtype, device=data[key].device)
            else:
                data[key] = img
        return data


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
        RemoveSmallObjectsPerClassd(keys=["pred"]),
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
            print_log=False),
        ])


@torch.inference_mode()
def run_inference(args, inference_config):
    # Load the model
    model = AttnUNet6(json.load(open('./model/model.json', 'r')))
    model.load_state_dict(torch.load('./model/model.pth', map_location='cpu', weights_only=True))
    model.eval().to(args.device)

    # Create dataset and dataloader
    spatial_tf, intensity_tf = get_pre_transforms(inference_config["pixdim"], 
                                                  inference_config["intensities"])
    post_tf = get_post_transforms(spatial_tf, args.output_dir)
    dataset = Dataset(
        data=get_image_files(args.input_dir), 
        transform=mt.Compose([spatial_tf, intensity_tf]))
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    for data in tqdm(dataset, desc="Inference"):
        data["pred"] = sliding_window_inference(
                    data["img"].to(args.device, non_blocking=True).unsqueeze(0),
                    roi_size=inference_config['shape'],
                    sw_batch_size=inference_config.get('sw_batch_size', 1),
                    predictor=lambda x: model(x),
                    overlap=inference_config.get('sw_overlap', 0.25),
                    mode="gaussian").cpu().squeeze(0)

        # Post-processing and saving results
        post_tf(data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=r'./inputs', help='dir of output')
    parser.add_argument('--output_dir', type=str, default=r'./outputs', help='dir of output')
    parser.add_argument('--device', type=str, default='cpu', help='device to run inference on')
    args = parser.parse_args()

    inference_config = json.load(open('./inference_config.json', 'r'))
    run_inference(args, inference_config)