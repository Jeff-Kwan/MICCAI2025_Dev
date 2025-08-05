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
import nibabel as nib


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


class RemoveSmallObjectsPerClass(Transform):
    def __init__(self,
            labels=list(range(1, 14)),
            min_sizes=[1e4, 1e3, 1e3, 1e3, 1e3, 1e3, 50, 100, 300, 100, 1000, 500, 500],
            connectivity=3):
        self.labels = labels
        self.min_sizes = min_sizes
        self.conn = connectivity

    def __call__(self, data):
        img = data.cpu().numpy()
        for lbl, ms in zip(self.labels, self.min_sizes):
            mask = (img == lbl)
            if mask.any():
                cleaned_mask = remove_small_objects(mask, min_size=ms, connectivity=self.conn)
                img[mask & (~cleaned_mask)] = 0
        return img 


@torch.inference_mode()
def run_inference(args, inference_config):
    # Load the model
    model = AttnUNet6(json.load(open('./model/model.json', 'r')))
    model.load_state_dict(torch.load('./model/model.pth', map_location='cpu', weights_only=True))
    model.eval().to(args.device)

    # Create dataset and dataloader
    upper, lower, mean, std = inference_config["intensities"]
    pixdim = inference_config["pixdim"]
    loader = mt.Compose([
        mt.LoadImaged(["img"], image_only=False, ensure_channel_first=True),
        mt.EnsureTyped(["img"], dtype=torch.float32, track_meta=True)])
    orientation = mt.Orientationd(["img"], axcodes="RAS", lazy=False)
    intensity = mt.Compose([
        mt.ThresholdIntensityd(["img"], above=False, threshold=upper, cval=upper),
        mt.ThresholdIntensityd(["img"], above=True, threshold=lower, cval=lower),
        mt.NormalizeIntensityd(["img"], subtrahend=mean, divisor=std)])
    saver = mt.SaveImaged(keys="pred",
            output_dir=args.output_dir, 
            output_postfix="", 
            output_ext=".nii.gz", 
            resample=False,     # Invert already resamples
            separate_folder=False,
            output_dtype=torch.uint8,
            print_log=False)
    dataset = Dataset(
        data=get_image_files(args.input_dir), 
        transform=mt.Compose([loader, orientation]))
    os.makedirs(args.output_dir, exist_ok=True)

    remove_small = RemoveSmallObjectsPerClass()
    centercrop = mt.CenterSpatialCropd(["img"], roi_size=inference_config["max_shape"])

    
    preprocess = mt.Compose([mt.Spacingd(["img"], pixdim=pixdim, mode="trilinear"), centercrop, intensity])
    invert_all = mt.Invertd(
            keys="pred",
            transform=mt.Compose([loader, orientation, preprocess]),
            orig_keys="img",
            meta_keys="pred_meta_dict",
            orig_meta_keys="img_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True)

    # Run inference
    for data in tqdm(dataset, desc="Inference"):
        # actual spatial size
        orig_pixdim = data["img"].meta["pixdim"][1:4].tolist()
        ss = [s*p1/p2 for s, p1, p2 in zip(data["img"].shape[1:], orig_pixdim, pixdim)]
        manual_invert = ss[0]*ss[1]*ss[2] > 2.5e7 or data["img"].numel() > 1.5e8

        if manual_invert:
            data["pred_meta_dict"] = data["img"].meta.copy()
            orig_axcodes = nib.orientations.aff2axcodes(data["img"].meta["original_affine"])
            invert_orientation = mt.Orientationd(keys=["pred"], axcodes=orig_axcodes)

            orig_shape = list(data["img"].shape)[1:]
            scale = [s / d for d, s in zip(pixdim, orig_pixdim)]
            data["img"] = torch.nn.functional.interpolate(
                data["img"].unsqueeze(0), 
                scale_factor=scale,
                mode="trilinear", align_corners=True).squeeze(0)
            full_shape = list(data["img"].shape)

            data = centercrop(data)
            data = intensity(data)

            pred = torch.argmax(sliding_window_inference(
                        data["img"].to(args.device, non_blocking=True).unsqueeze(0),
                        roi_size=inference_config['shape'],
                        sw_batch_size=inference_config.get('sw_batch_size', 1),
                        predictor=lambda x: model(x),
                        overlap=inference_config.get('sw_overlap', 0.25),
                        mode="gaussian").cpu().squeeze(0), dim=0, keepdim=True).to(torch.uint8)

            # Remove small objects
            pred = torch.tensor(remove_small(pred))
            
            # Invert center crop
            data["pred"] = torch.zeros(full_shape, dtype=torch.uint8)
            start = [(s - p) // 2 for s, p in zip(full_shape[1:], pred.shape[1:])]
            end = [start[i] + pred.shape[i+1] for i in range(len(start))]
            data["pred"][:, *tuple(slice(s, e) for s, e in zip(start, end))] = pred

            # Invert spacing
            data["pred"] = torch.nn.functional.interpolate(
                data["pred"].unsqueeze(0), 
                size=orig_shape,
                mode="nearest").squeeze(0)
            
            # Invert orientation
            data["pred"] = MetaTensor(data["pred"], meta=data["pred_meta_dict"])
            data = invert_orientation(data)
            
            # Post-processing and saving results
            saver(data)

        else:
            data = preprocess(data)
            data["pred"] = torch.argmax(sliding_window_inference(
                        data["img"].to(args.device, non_blocking=True).unsqueeze(0),
                        roi_size=inference_config['shape'],
                        sw_batch_size=inference_config.get('sw_batch_size', 1),
                        predictor=lambda x: model(x),
                        overlap=inference_config.get('sw_overlap', 0.25),
                        mode="gaussian").cpu().squeeze(0), dim=0, keepdim=True).to(torch.uint8)
            data["pred"] = remove_small(data["pred"])

            data["pred_meta_dict"] = data["img"].meta.copy()
            data["pred"] = MetaTensor(data["pred"], meta=data["pred_meta_dict"])
            data = invert_all(data)
            saver(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=r'./inputs', help='dir of output')
    parser.add_argument('--output_dir', type=str, default=r'./outputs', help='dir of output')
    parser.add_argument('--device', type=str, default='cpu', help='device to run inference on')
    args = parser.parse_args()

    inference_config = json.load(open('./inference_config.json', 'r'))
    run_inference(args, inference_config)