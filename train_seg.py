import os
import json
import torch
import numpy as np
from datetime import datetime
from torch.optim import AdamW, lr_scheduler
from monai.data import PersistentDataset, DataLoader
from monai.losses import DiceCELoss

from utils import Trainer, get_transforms, get_data_files
from model.Harmonics import HarmonicSeg

# For use of PersistentDataset
torch.serialization.add_safe_globals([np.dtype, np.dtypes.Int64DType,
                        np.ndarray, np.core.multiarray._reconstruct])


def training(model_params, train_params, output_dir, comments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%H-%M")
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join('output', date_str, f'{timestamp}-{output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    # Data loading
    train_transform, val_transform = get_transforms(train_params['shape'],
                                train_params['norm_clip'], 
                                train_params['pixdim'])

    # Persistent dataset needs list of file paths?
    train_dataset = PersistentDataset(
        data = get_data_files(
            images_dir="data/FLARE-Task2-LaptopSeg/train_gt_label/imagesTr",
            labels_dir="data/FLARE-Task2-LaptopSeg/train_gt_label/labelsTr"),
        transform=train_transform,
        cache_dir="data/cache/gt_label")
    val_dataset = PersistentDataset(
        data = get_data_files(
            images_dir="data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Images",
            labels_dir="data/FLARE-Task2-LaptopSeg/validation/Validation-Public-Labels"),
        transform=val_transform,
        cache_dir="data/cache/val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params['batch_size'],
        shuffle=True,
        num_workers=30,
        pin_memory=True,
        persistent_workers=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=30)


    # Training setup
    model = HarmonicSeg(model_params)
    optimizer = AdamW(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['epochs'])
    # criterion = DiceFocalLoss(softmax=True, include_background=True, to_onehot_y=True,
    #             weight=torch.tensor([0.05] + [1.0] * (train_params['num_classes'] - 1), device=device))
    criterion = DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        weight=torch.tensor([0.05] + [1.0] * (train_params['num_classes'] - 1), device=device),
        label_smoothing=0.1
    )

    # Compilation acceleration
    if train_params.get('compile', False):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')
        # model = torch.compile(model)

    # Trainer
    trainer = Trainer(model, optimizer, criterion, scheduler, 
                      train_params, output_dir, device, comments)
    trainer.train(train_loader, val_loader)



if __name__ == "__main__":
    model_params = json.load(open("configs/model/base.json"))

    train_params = {
        'epochs': 200,
        'batch_size': 4,
        'aggregation': 1,
        'learning_rate': 1e-3,
        'weight_decay': 2e-2,
        'num_classes': 14,
        'shape': (192, 192, 192),
        'norm_clip': (-200, 400, -1.0, 1.0),
        'pixdim': (1.0, 1.0, 1.0),
        'compile': True,
        'autocast': False,
        'sw_batch_size': 64,
        'sw_overlap': 0.2
    }

    output_dir = "HarmonicSeg-50GT"
    comments = ["HarmonicSeg - 50 Gound Truth set training"]

    training(model_params, train_params, output_dir, comments)