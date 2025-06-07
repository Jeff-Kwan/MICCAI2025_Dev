import os
import torch
from torch.optim import AdamW, lr_scheduler
from monai.data import PersistentDataset, ThreadDataLoader
from monai.losses import DiceFocalLoss
import monai.transforms as mt
from utils import Trainer
from model.Harmonics import HarmonicSeg

def get_transforms(device, shape, norm_clip, pixdim):
    train_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.ScaleIntensityRanged(
                keys=["image"],
                a_min=norm_clip[0],
                a_max=norm_clip[1],
                b_min=norm_clip[2],
                b_max=norm_clip[3],
                clip=True,
            ),
            mt.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"),
            mt.Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            mt.EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            mt.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=shape,
                pos=1,
                neg=1,
                image_key="image",
                image_threshold=0,
            ),
            mt.RandAxisFlip(
                prob=0.10,
            ),
            mt.RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            mt.RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            # mt.RandAffined(
            #     keys=["image", "label"],
            #     mode=("bilinear", "nearest"),
            #     prob=0.5,
            #     rotate_range=(0.1, 0.1, 0.1),
            #     scale_range=(0.2, 0.2, 0.2),
            #     padding_mode="zeros"),
            mt.RandGaussianNoised(
                keys=["image"],
                prob=0.1,
                mean=0.0,
                std=0.05,
            ),
        ]
    )
    val_transform = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            mt.ScaleIntensityRanged(keys=["image"],
                                    a_min=norm_clip[0],
                                    a_max=norm_clip[1],
                                    b_min=norm_clip[2],
                                    b_max=norm_clip[3],
                                    clip=True),
            mt.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"),
            mt.Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            mt.EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )
    return train_transform, val_transform

def training(model_params, train_params, output_dir, comments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Data loading
    transforms = get_transforms(device, model_params['shape'])

    # Persistent dataset needs list of file paths?


    # Training setup
    model = HarmonicSeg(model_params)
    optimizer = AdamW(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['epochs'])
    criterion = DiceFocalLoss(softmax=True, include_background=True)

    # Trainer
    trainer = Trainer(model, optimizer, criterion, scheduler, 
                      train_params, output_dir, device, comments)
    