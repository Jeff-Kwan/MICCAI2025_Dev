import os
import json
import torch
import numpy as np
from datetime import datetime
from torch.optim import AdamW, lr_scheduler
from monai.data import PersistentDataset, ThreadDataLoader, Dataset, meta_tensor
from monai.losses import DiceFocalLoss
from monai.utils.enums import MetaKeys, SpaceKeys, TraceKeys

from utils import Trainer, get_transforms, get_data_files
from model.Harmonics import HarmonicSeg

# For use of PersistentDataset
torch.serialization.add_safe_globals([np.dtype, np.ndarray, np.core.multiarray._reconstruct, 
    np.dtypes.Int64DType, np.dtypes.Int32DType, np.dtypes.Int16DType, np.dtypes.UInt8DType,
    np.dtypes.Float32DType, np.dtypes.Float64DType,
    meta_tensor.MetaTensor, MetaKeys, SpaceKeys, TraceKeys])

def training(model_params, train_params, output_dir, comments, pretrained_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%H-%M")
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join('output', date_str, f'{timestamp}-{output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    # Data loading
    train_transform, val_transform = get_transforms(train_params['shape'],
                                train_params['num_crops'])

    # Persistent dataset needs list of file paths?
    # train_dataset = PersistentDataset(
    #     data = get_data_files(
    #         images_dir="data/preprocessed/train_gt/images",
    #         labels_dir="data/preprocessed/train_gt/labels"),
    #     transform=train_transform,
    #     cache_dir="data/cache/gt_label")
    train_dataset = Dataset(
        data=# Combine both pseudo-label datasets
            get_data_files(
            images_dir="data/preprocessed/train_pseudo/images",
            labels_dir="data/preprocessed/train_pseudo/aladdin5")\
            +\
            get_data_files(
            images_dir="data/preprocessed/train_pseudo/images",
            labels_dir="data/preprocessed/train_pseudo/blackbean"),
        transform=train_transform)
    val_dataset = PersistentDataset(
        data = get_data_files(
            images_dir="data/preprocessed/val/images",
            labels_dir="data/preprocessed/val/labels"),
        transform=val_transform,
        cache_dir="data/cache/val")

    train_loader = ThreadDataLoader(
        train_dataset,
        batch_size=train_params['batch_size'],
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True)
    val_loader = ThreadDataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=32,
        persistent_workers=False)


    # Training setup
    model = HarmonicSeg(model_params)
    optimizer = AdamW(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['epochs'])
    criterion = DiceFocalLoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        weight=torch.tensor([0.02] + [1.0] * 13, device=device))

    # Load pretrained weights
    print(f'Loading pretrained weights from {pretrained_path}')
    raw = torch.load(pretrained_path)
    state_dict = raw.get('state_dict', raw)
    # Strip the "_orig_mod." prefix from any key that has it
    clean_state_dict = {
        (k[len("_orig_mod."): ] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()}
    # if key contains self.out_conv, remove it (different output channels)
    clean_state_dict = {k: v for k, v in clean_state_dict.items() if 'out_conv' not in k}
    model.load_state_dict(clean_state_dict, strict=False)

    # Compilation acceleration
    if train_params.get('compile', False):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')
        model = torch.compile(model)

    # First, only train output layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.out_conv.parameters():
        param.requires_grad = True

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
    pretrained_path = "output/2025-06-08/09-49-MIM-2000-96x3/model.pth"

    train_params = {
        'epochs': 10,
        'batch_size': 1,
        'aggregation': 2,
        'learning_rate': 1e-3,
        'weight_decay': 1e-2,
        'num_classes': 14,
        'shape': (96, 96, 96),
        'num_crops': 8,
        'compile': True,
        'autocast': True,
        'sw_batch_size': 64,
        'sw_overlap': 0.1
    }
    torch._dynamo.config.cache_size_limit = 16  # Up the cache size limit for dynamo

    output_dir = "Transfer-2000-GT50-96x3"
    comments = ["HarmonicSeg output layer - 2000 MIM -> 50GT Segmentation 10 epochs",
        "(96, 96, 96) shape", 
        "DiceFocal"]

    training(model_params, train_params, output_dir, comments, pretrained_path)