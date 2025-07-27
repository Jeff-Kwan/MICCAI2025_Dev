import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # Fragmentation
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback
from monai.data import DataLoader, Dataset
from monai.losses import DiceFocalLoss

from utils.dataset import get_transforms, get_data_files
from model.AttnUNet2 import AttnUNet
from utils.ddp_trainer import DDPTrainer

torch.multiprocessing.set_sharing_strategy('file_system')

def main_worker(rank: int,
                world_size: int,
                model_path: str,
                model_params: dict,
                train_params: dict,
                comments: list):
    """
    Entry point for each spawned process.
    """
    try:
        # 1) Set the GPU device for this rank
        torch.cuda.set_device(rank)

        # 2) Initialize the process group
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:29500',
            world_size=world_size,
            rank=rank
        )

        # Datasets
        _, val_tf = get_transforms(
            train_params['shape'],
            train_params['data_augmentation']['spatial'],
            train_params['data_augmentation']['intensity'],
            train_params['data_augmentation']['coarse'])
        val_ds = Dataset(
            data=get_data_files(
                images_dir="data/preprocessed/val/images",
                labels_dir="data/preprocessed/val/labels",
                extension='.npy'),
            transform=val_tf)
        val_sampler = torch.utils.data.DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            sampler=val_sampler,
            num_workers=16,
            pin_memory=True)

        # Model, optimizer, scheduler, loss
        model = AttnUNet(model_params)
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        criterion = DiceFocalLoss(
            include_background=True, 
            to_onehot_y=True, 
            softmax=True, 
            weight=torch.tensor([0.1, 2.9, 5.0, 4.8, 5.7, 5.7, 5.8, 8.8, 
                                 8.6, 6.7, 7.5, 4.4, 5.9, 5.0], device=rank),
            lambda_focal=1,
            lambda_dice=1,)


        # Initialize trainer and start
        trainer = DDPTrainer(
            model=model,
            optimizer=None,
            criterion=criterion,
            scheduler=None,
            train_params=train_params,
            output_dir=None,
            local_rank=rank,
            world_size=world_size,
            comments=comments)
        loss, metrics = trainer.evaluate(val_loader)
        if rank == 0:
            print(f"Validation Loss: {loss:.4f}")
            print("Dice:", metrics['dice'])

    except Exception as e:
        print(f"Rank {rank} crashed:", traceback.format_exc())
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # If needed:    pkill -f -- '--multiprocessing-fork'
    # Load configs
    model_params = json.load(open("configs/model/attn_unet.json"))
    model_path = "output/2025-06-19/15-27-AttnUNet/model.pth"
    train_params = {
        'epochs': 300,
        'batch_size': 1,    # effectively x4
        'aggregation': 1,
        'learning_rate': 2e-4,
        'weight_decay': 1e-2,
        'num_classes': 14,
        'shape': (256, 192, 128),
        'compile': False,
        'autocast': False,
        'sw_batch_size': 2,
        'sw_overlap': 1/4,
        'data_augmentation': {
            # [I, Affine, Flip, Rotate90, Elastic]
            'spatial': [2, 2, 1, 1, 1],  
            # [I, Smooth, Noise, Bias, Contrast, Sharpen, Histogram]
            'intensity': [3, 2, 1, 1, 1, 1, 1],  
            # [I, Dropout, Shuffle]
            'coarse': [2, 1, 1]  
        }
    }
    comments = []

    gpu_count = torch.cuda.device_count()
    try:
        mp.spawn(
            main_worker,
            args=(gpu_count, model_path, model_params, train_params,  comments),
            nprocs=gpu_count,
            join=True)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main process. Terminating children...")
        mp.get_context('spawn')._shutdown()
