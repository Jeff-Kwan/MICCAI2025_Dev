import os
from pathlib import Path
from tqdm import tqdm
import torch
import monai.transforms as mt
from monai.data import Dataset, ThreadDataLoader
from monai.config import KeysCollection

class NormalizeChannelSumd(mt.MapTransform):
    """
    Dictionary-based transform that normalizes the channel dimension so that
    the sum across channels is 1 for each voxel location.
    Assumes input tensors are in channel-first format: (C, H, W[, D]).
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            # Compute sum across channels at each voxel location
            sum_across_channels = img.sum(dim=0, keepdim=True)
            # Avoid division by zero
            sum_across_channels[sum_across_channels == 0] = 1.0
            # Normalize each channel
            d[key] = img / sum_across_channels
        return d


def get_pseudo_data(aladdin, blackbean, extension=".nii.gz"):
    aladdin = Path(aladdin)
    blackbean = Path(blackbean)
    data = []
    for img_path in aladdin.glob(f"*{extension}"):
        aladdin_path = aladdin / img_path.name
        blackbean_path = blackbean / img_path.name
        assert blackbean_path.exists(), f"Match for {img_path.name} not found in {blackbean}"
        data.append({
            "aladdin": str(aladdin_path),
            "blackbean": str(blackbean_path)
        })
    print(f"[INFO] found {len(data)} image/label data")
    return data



def process_dataset(aladdin, blackbean, out_dir, pixdim):
    # create output dirs
    os.makedirs(out_dir, exist_ok=True)

    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["aladdin", "blackbean"], ensure_channel_first=True),
            mt.Orientationd(keys=["aladdin", "blackbean"], axcodes="RAS", lazy=True),
            mt.Spacingd(
                keys=["aladdin", "blackbean"],
                pixdim=pixdim,
                mode="nearest",
                lazy=True),
            mt.ThresholdIntensityd(
                keys=["aladdin", "blackbean"],
                above=False,
                threshold=14,   # 14 classes
                cval=0),
            mt.AsDiscreted(
                keys=["aladdin", "blackbean"],
                to_onehot=14),  # 14 classes
            mt.EnsureTyped(
                keys=["aladdin", "blackbean"],
                dtype=torch.float32,
                track_meta=True),
            mt.MeanEnsembled(
                keys=["aladdin", "blackbean"],
                output_key="label"),
            mt.DeleteItemsd(
                keys=["aladdin", "blackbean"]),
            NormalizeChannelSumd(keys=["label"]),
            mt.SaveImaged(
                keys=["label"],
                output_dir=out_dir,
                output_postfix="",
                output_ext=".nii.gz",
                separate_folder=False,
                output_dtype=torch.float32,
                print_log=False),
            mt.DeleteItemsd(keys=["label"])
        ]
    )

    # build the MONAI dataset
    dataset = Dataset(data=get_pseudo_data(aladdin, blackbean), transform=transform)
    dataloader = ThreadDataLoader(
        dataset,
        batch_size=1,
        num_workers=36,
        prefetch_factor=8,
    )

    # iterate, transform, and save
    for batch in tqdm(dataloader, desc=f"Creating Soft Labels"):
        pass

    return 



def process_gt(in_dir, out_dir, pixdim):
    # create output dirs
    os.makedirs(out_dir, exist_ok=True)

    transform = mt.Compose(
        [
            mt.LoadImaged(keys=["label"], ensure_channel_first=True),
            mt.AsDiscreted(
                keys=["label"],
                to_onehot=14),  # 14 classes
            mt.EnsureTyped(
                keys=["label"],
                dtype=torch.float32,
                track_meta=True),
            mt.Orientationd(keys=["label"], axcodes="RAS", lazy=True),
            mt.Spacingd(
                keys=["label"],
                pixdim=pixdim,
                mode="trilinear",
                lazy=True),
            NormalizeChannelSumd(keys=["label"]),
            mt.SaveImaged(
                keys=["label"],
                output_dir=out_dir,
                output_postfix="",
                output_ext=".nii.gz",
                separate_folder=False,
                output_dtype=torch.float32,
                print_log=False),
            mt.DeleteItemsd(keys=["label"])
        ]
    )

    # build the MONAI dataset
    dir = Path(in_dir)
    if not dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir!r}")
    names = sorted(
        entry.name
        for entry in os.scandir(dir)
        if entry.is_file() and entry.name.endswith(".nii.gz"))
    dataset = Dataset(data=[{"label": str(dir / name)} for name in names], transform=transform)
    dataloader = ThreadDataLoader(
        dataset,
        batch_size=1,
        num_workers=50)

    # iterate, transform, and save
    for batch in tqdm(dataloader, desc=f"Processing GT to Soft"):
        pass

    return 


if __name__ == "__main__":
    pixdim = (0.8, 0.8, 2.5)
    # process_gt(
    #     "data/FLARE-Task2-LaptopSeg/train_gt_label/labelsTr",
    #     "data/nifti/train_gt/softlabel",
    #     pixdim)
    process_dataset(
        "data/FLARE-Task2-LaptopSeg/train_pseudo_label/flare22_aladdin5_pseudo",
        "data/FLARE-Task2-LaptopSeg/train_pseudo_label/pseudo_label_blackbean_flare22",
        "data/nifti/train_pseudo/softlabel",
        pixdim)