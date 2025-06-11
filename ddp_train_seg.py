import os
import json
import torch
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

def main_worker(local_rank, world_size, model_params, train_params, output_dir, comments):
    # 1) set device for this process
    torch.cuda.set_device(local_rank)

    # 2) accel-provided init
    dist.init_process_group(backend='nccl', init_method='env://')

    # 3) only rank 0 creates the output folder
    if local_rank == 0:
        timestamp = datetime.now().strftime("%H-%M")
        date_str  = datetime.now().strftime("%Y-%m-%d")
        full_output = os.path.join('output', date_str, f'{timestamp}-{output_dir}')
        os.makedirs(full_output, exist_ok=True)
    else:
        full_output = None

    # Data transforms
    train_tf, val_tf = get_transforms(train_params['shape'], train_params['num_crops'])

    # Datasets
    train_ds = PersistentDataset(
        data=get_data_files(
            images_dir="data/preprocessed/train_gt/images",
            labels_dir="data/preprocessed/train_gt/labels"),
        transform=train_tf,
        cache_dir="data/cache/gt_label")
    val_ds = PersistentDataset(
        data=get_data_files(
            images_dir="data/preprocessed/val/images",
            labels_dir="data/preprocessed/val/labels"),
        transform=val_tf,
        cache_dir="data/cache/val")

    # Distributed sampler & loader
    train_sampler = torch.utils.data.DistributedSampler(
        train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = torch.utils.data.DistributedSampler(
        val_ds, num_replicas=world_size, rank=local_rank, shuffle=False)
    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=train_params['batch_size'],
        sampler=train_sampler,
        num_workers=88,
        prefetch_factor=1,
        pin_memory=True,
        persistent_workers=True)
    val_loader = ThreadDataLoader(
        val_ds,
        batch_size=4,
        sampler=val_sampler,
        num_workers=32,
        persistent_workers=False)

    # Model, optimizer, scheduler, loss
    model = HarmonicSeg(model_params)
    optimizer = AdamW(
        model.parameters(),
        lr=train_params['learning_rate'],
        weight_decay=train_params['weight_decay'])
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_params['epochs'])
    criterion = DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        weight=torch.tensor([0.01] + [1.0]*13, device=f"cuda:{local_rank}"),
        label_smoothing=0.1,
        lambda_ce=0.34,
        lambda_dice=0.66)

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
        output_dir=full_output,
        local_rank=local_rank,
        world_size=world_size,
        comments=comments)
    trainer.train(train_loader, val_loader)

    # Rank 0 does final evaluation
    if local_rank == 0:
        test_loss, test_metrics = trainer.evaluate(val_loader)
        with open(os.path.join(full_output, 'results.txt'), 'a') as f:
            f.write(f"\nLast Model Test: Loss={test_loss:.5f}, Dice={test_metrics['dice']:.5f}\n")
        print(f"[Last] Test Loss: {test_loss:.5f}, Dice: {test_metrics['dice']:.5f}")

        trainer.model.load_state_dict(torch.load(os.path.join(full_output, 'best_model.pth')))
        best_loss, best_metrics = trainer.evaluate(val_loader)
        with open(os.path.join(full_output, 'results.txt'), 'a') as f:
            f.write(f"Best Model Test: Loss={best_loss:.5f}, Dice={best_metrics['dice']:.5f}\n")
        print(f"[Best] Test Loss: {best_loss:.5f}, Dice: {best_metrics['dice']:.5f}")

    # teardown
    dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    '''
    torchrun --nproc_per_node=4 --master_port=29500 ddp_train_seg.py
    '''

    parser = argparse.ArgumentParser(description="DDP Training with torchrun")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Load configs
    model_params = json.load(open("configs/model/large.json"))
    train_params = {
        'epochs': 100,
        'batch_size': 1,
        'aggregation': 2,
        'learning_rate': 3e-4,
        'weight_decay': 1e-2,
        'num_classes': 14,
        'shape': (160, 160, 80),
        'num_crops': 8,
        'compile': True,
        'autocast': True,
        'sw_batch_size': 16,
        'sw_overlap': 1/8
    }
    output_dir = "PseudolabelsAll"
    comments = [
        "HarmonicSeg Base - 2000 Aladdin Pseudolabels training",
        "DiceCE, 8-sample rand crop + fewer augmentations",
        "Spatial [1, 1, 0, 0, 1]; Intensity [3, 1, 1, 0, 1, 0, 0]; Coarse [2, 1, 1]"
    ]

    main_worker(local_rank, world_size, model_params, train_params, output_dir, comments)
