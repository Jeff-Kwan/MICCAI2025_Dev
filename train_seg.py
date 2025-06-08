import os
import json
import torch
import numpy as np
from datetime import datetime
from torch.optim import AdamW, lr_scheduler
from monai.data import PersistentDataset, DataLoader, Dataset, meta_tensor
from monai.losses import DiceCELoss
from monai.utils.enums import MetaKeys, SpaceKeys, TraceKeys

from utils import Trainer, get_transforms, get_data_files
from model.Harmonics import HarmonicSeg

# For use of PersistentDataset
torch.serialization.add_safe_globals([np.dtype, np.ndarray, np.core.multiarray._reconstruct, 
    np.dtypes.Int64DType, np.dtypes.Int32DType, np.dtypes.Int16DType, np.dtypes.UInt8DType,
    np.dtypes.Float32DType, np.dtypes.Float64DType,
    meta_tensor.MetaTensor, MetaKeys, SpaceKeys, TraceKeys])


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
    # train_dataset = PersistentDataset(
    #     data = get_data_files(
    #         images_dir="data/FLARE-Task2-LaptopSeg/train_gt_label/imagesTr",
    #         labels_dir="data/FLARE-Task2-LaptopSeg/train_gt_label/labelsTr"),
    #     transform=train_transform,
    #     cache_dir="data/cache/gt_label")
    train_dataset = Dataset(
        data=# Combine both pseudo-label datasets
            get_data_files(
            images_dir="data/FLARE-Task2-LaptopSeg/train_pseudo_label/imagesTr",
            labels_dir="data/FLARE-Task2-LaptopSeg/train_pseudo_label/flare22_aladdin5_pseudo")\
            +\
            get_data_files(
            images_dir="data/FLARE-Task2-LaptopSeg/train_pseudo_label/imagesTr",
            labels_dir="data/FLARE-Task2-LaptopSeg/train_pseudo_label/pseudo_label_blackbean_flare22"),
        transform=train_transform)
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
        num_workers=64,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=32,
        persistent_workers=False)


    # Training setup
    model = HarmonicSeg(model_params)
    optimizer = AdamW(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['epochs'])
    criterion = DiceCELoss(
        include_background=False, 
        to_onehot_y=True, 
        softmax=True, 
        weight=torch.tensor([0.01] + [1.0] * 13, device=device),
        label_smoothing=0.2,
        lambda_ce=0.4,
        lambda_dice=0.6,)

    # Compilation acceleration
    if train_params.get('compile', False):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')
        model = torch.compile(model)

    # Trainer
    trainer = Trainer(model, optimizer, criterion, scheduler, 
                      train_params, output_dir, device, comments)
    trainer.train(train_loader, val_loader)

    # Test Last Model
    test_loss, test_metrics = trainer.evaluate(val_loader)
    test_results = {'test_loss': test_loss}
    test_results.update(test_metrics)
    with open(f'{output_dir}/results.txt', 'a') as f:
        f.write(f'\nLast Model Test Performance:\n{json.dumps(test_results, indent=4)}')
    print(f'Last Model Performance - Test Loss: {test_loss:.5f}, Dice: {test_metrics["dice"]:.5f}')
    
    # Test Best Model
    trainer.model.load_state_dict(torch.load(f'{output_dir}/best_model.pth', weights_only=True))
    test_loss, test_metrics = trainer.evaluate(val_loader)
    test_results = {'test_loss': test_loss}
    test_results.update(test_metrics)
    with open(f'{output_dir}/results.txt', 'a') as f:
        f.write(f'\nBest Model Test Performance:\n{json.dumps(test_results, indent=4)}')
    print(f'Best Model Performance - Test Loss: {test_loss:.5f}, Dice: {test_metrics["dice"]:.5f}')



if __name__ == "__main__":
    model_params = json.load(open("configs/model/base.json"))

    train_params = {
        'epochs': 50,
        'batch_size': 2,
        'aggregation': 4,
        'learning_rate': 1e-3,
        'weight_decay': 5e-2,
        'num_classes': 14,
        'shape': (96, 96, 96),
        'norm_clip': (-325, 325, -1.0, 1.0),
        'pixdim': (1.5, 1.5, 1.5),
        'compile': True,
        'autocast': True,
        'sw_batch_size': 64,
        'sw_overlap': 0.1
    }
    torch._dynamo.config.cache_size_limit = 16  # Up the cache size limit for dynamo

    output_dir = "PseudolabelsAll-96x3"
    comments = ["HarmonicSeg - 2000 x2 Pseudolabels training",
        "(96, 96, 96) shape, 1.5mm pixdim for faster training?", 
        "DiceCE, 16-sample rand crop + a lot of augmentations"]

    training(model_params, train_params, output_dir, comments)