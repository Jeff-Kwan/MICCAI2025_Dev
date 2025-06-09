import os
from tqdm import tqdm
import numpy as np
import torch
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader
import multiprocessing as mp

def get_file_paths(dir, ext=".nii.gz"):
    return {
        "image": [os.path.join(dir, f)
                  for f in sorted(os.listdir(dir)) if f.endswith(ext)],
    }


def process_dataset(images_dir, labels_dir, out_image_dir, out_label_dir, pixdim):
    # define the validation transform
    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image"], ensure_channel_first=True),
            mt.EnsureTyped(
                keys=["image"],
                dtype=[torch.float32],
                track_meta=True,
            ),
            mt.ThresholdIntensityd( # upper bound 99.5%
                keys=["image"],
                above=True,
                threshold=1000.0,
                cval=1000,
            ),
            mt.ThresholdIntensityd( # lower bound 0.5%
                keys=["image"],
                above=False,
                threshold=-1000.0, 
                cval=-1000,
            ),
            mt.NormalizeIntensityd(
                keys=["image"],
                subtrahend=0.0,
                divisor=1.0,
            ),
            mt.SaveImaged(
                keys=["image"],
                output_dir=out_image_dir,
                output_postfix="",
                output_ext=".nii.gz",
                separate_folder=False,
                output_dtype=torch.float32,
                print_log=False,
            ),
        ]
    )

    # build the MONAI dataset
    dataset = Dataset(data=get_file_paths(images_dir)
                      , transform=transform)
    dataloader = ThreadDataLoader(
        dataset,
        batch_size=1,
        num_workers=64,
    )


    # iterate, transform, and save
    for batch in tqdm(dataloader, desc="Processing images"):
        pass




if __name__ == "__main__":
    pass
