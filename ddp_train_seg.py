# ddp_training.py

import os
import json
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from datetime import datetime
from torch.optim import AdamW, lr_scheduler
from monai.data import PersistentDataset, ThreadDataLoader
from monai.losses import DiceCELoss
from monai.utils.enums import MetaKeys, SpaceKeys, TraceKeys

from utils import get_transforms, get_data_files
from model.Harmonics import HarmonicSeg
from utils.ddp_trainer import DDPTrainer

# For PersistentDataset pickling
torch.serialization.add_safe_globals([
    np.dtype, np.ndarray, np.core.multiarray._reconstruct,
    np.dtypes.Int64DType, np.dtypes.Int32DType, np.dtypes.Int16DType,
    np.dtypes.UInt8DType, np.dtypes.Float32DType, np.dtypes.Float64DType,
    MetaKeys, SpaceKeys, TraceKeys
])

def main_worker(local_rank, world_size, model_params, train_params, base_output, comments):
    # Construct unique output dir per run (only on rank 0)
    if local_rank == 0:
        timestamp = datetime.now().strftime("%H-%M")
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_dir = os.path.join(base_output, date_str, timestamp)
    else:
        output_dir = None
    # Broadcast the output path string to all ranks
    output_dir = dist.broadcast_object_list([output_dir], src=0)[0]

    # Data transforms
    train_tf, val_tf = get_transforms(train_params['shape'],
                                      train_params['num_crops'],
                                      device=f"cuda:{local_rank}")

    # Datasets
    train_ds = PersistentDataset(
        data=get_data_files("data/preprocessed/train_pseudo/images",
                            "data/preprocessed/train_pseudo/aladdin5"),
        transform=train_tf,
        cache_dir="data/cache/pseudo_label"
    )
    val_ds = PersistentDataset(
        data=get_data_files("data/preprocessed/val/images",
                            "data/preprocessed/val/labels"),
        transform=val_tf,
        cache_dir="data/cache/val"
    )

    # Distributed sampler & loader
    train_sampler = torch.utils.data.DistributedSampler(
        train_ds, num_replicas=world_size, rank=local_rank, shuffle=True
    )
    val_sampler = torch.utils.data.DistributedSampler(
        val_ds, num_replicas=world_size, rank=local_rank, shuffle=False
    )
    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=train_params['batch_size'],
        sampler=train_sampler,
        num_workers=24,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = ThreadDataLoader(
        val_ds,
        batch_size=4,
        sampler=val_sampler,
        shuffle=False,
        num_workers=24,
        persistent_workers=False
    )

    # Model, optimizer, scheduler, loss
    model = HarmonicSeg(model_params)
    optimizer = AdamW(
        model.parameters(),
        lr=train_params['learning_rate'],
        weight_decay=train_params['weight_decay']
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_params['epochs']
    )
    criterion = DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        weight=torch.tensor([0.01] + [1.0]*13, device=f"cuda:{local_rank}"),
        label_smoothing=0.1,
        lambda_ce=0.34,
        lambda_dice=0.66,
    )

    # Optional compile
    if train_params.get('compile', False):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')
        model = torch.compile(model)

    # Initialize and run DDP trainer
    trainer = DDPTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_params=train_params,
        output_dir=output_dir,
        local_rank=local_rank,
        world_size=world_size,
        comments=comments
    )
    trainer.train(train_loader, val_loader)

    # On rank 0, test last and best
    if local_rank == 0:
        test_loss, test_metrics = trainer.evaluate(val_loader)
        with open(os.path.join(output_dir, 'results.txt'), 'a') as f:
            f.write(f"\nLast Model Test: Loss={test_loss:.5f}, Dice={test_metrics['dice']:.5f}\n")
        print(f"[Last] Test Loss: {test_loss:.5f}, Dice: {test_metrics['dice']:.5f}")

        # Best
        trainer.model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
        best_loss, best_metrics = trainer.evaluate(val_loader)
        with open(os.path.join(output_dir, 'results.txt'), 'a') as f:
            f.write(f"Best Model Test: Loss={best_loss:.5f}, Dice={best_metrics['dice']:.5f}\n")
        print(f"[Best] Test Loss: {best_loss:.5f}, Dice: {best_metrics['dice']:.5f}")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    # Number of GPUs
    world_size = torch.cuda.device_count()

    # Load configs
    model_params = json.load(open("configs/model/base.json"))
    train_params = {
        'epochs': 100,
        'batch_size': 4,
        'aggregation': 2,
        'learning_rate': 3e-4,
        'weight_decay': 1e-2,
        'num_classes': 14,
        'shape': (128, 128, 128),
        'num_crops': 8,
        'compile': True,
        'autocast': True,
        'sw_batch_size': 32,
        'sw_overlap': 1/8
    }
    base_output = "output"
    comments = [
        "HarmonicSeg Base - 2000 Aladdin Pseudolabels training",
        "(128, 128, 128) shape",
        "DiceCE, 8-sample rand crop + heavy augmentations"
    ]

    mp.spawn(
        main_worker,
        args=(world_size, model_params, train_params, base_output, comments),
        nprocs=world_size,
        join=True
    )
