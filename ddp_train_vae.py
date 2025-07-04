import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Fragmentation
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from torch.optim import AdamW, lr_scheduler
from monai.data import DataLoader, Dataset
from monai.losses import DiceFocalLoss
import traceback

from utils.dataset import get_vae_transforms, get_data_files
from model.VAEDual import VAEPosterior
from utils.vae_trainer import VAETrainer

torch.multiprocessing.set_sharing_strategy('file_system')

def main_worker(rank: int,
                world_size: int,
                model_params: dict,
                train_params: dict,
                output_dir: str,
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

        # 3) Only rank 0 creates output folder
        if rank == 0:
            timestamp = datetime.now().strftime("%H-%M")
            date_str  = datetime.now().strftime("%Y-%m-%d")
            full_output = os.path.join('output', date_str, f'{timestamp}-{output_dir}')
            os.makedirs(full_output, exist_ok=True)
        else:
            full_output = None

        # Datasets
        train_tf, val_tf = get_vae_transforms(
            train_params['shape'],
            train_params['data_augmentation']['spatial'],
            train_params['data_augmentation']['intensity'],
            train_params['data_augmentation']['coarse'])
        train_ds = Dataset(
            data=get_data_files(
                images_dir="data/small/train_gt/images",
                labels_dir="data/small/train_gt/labels",
                extension='.npy') * 2 \
            + get_data_files(
                images_dir="data/small/train_pseudo/images",
                labels_dir="data/small/train_pseudo/pseudo",
                extension='.npy'),
            transform=train_tf)
        val_ds = Dataset(
            data=get_data_files(
                images_dir="data/small/val/images",
                labels_dir="data/small/val/labels",
                extension='.npy'),
            transform=val_tf)
        train_sampler = torch.utils.data.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = torch.utils.data.DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=train_params['batch_size'],
            sampler=train_sampler,
            num_workers=42,
            pin_memory=True,
            persistent_workers=True)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=False,
            persistent_workers=False)


        # Model, optimizer, scheduler, loss
        model = VAEPosterior(model_params)
        optimizer = AdamW(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['epochs'])
        criterion = DiceFocalLoss(
            include_background=True, 
            to_onehot_y=True, 
            softmax=True, 
            weight=torch.tensor([0.01] + [1.0] * 13, device=rank),
            lambda_focal=1,
            lambda_dice=1,)

        # Initialize trainer and start
        trainer = VAETrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            train_params=train_params,
            output_dir=full_output,
            local_rank=rank,
            world_size=world_size,
            comments=comments)
        trainer.train(train_loader, val_loader)

    except Exception as e:
        print(f"Rank {rank} crashed:", traceback.format_exc(), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # If needed:    pkill -f -- '--multiprocessing-fork'
    # Load configs
    model_params = json.load(open("configs/model/vae.json"))
    train_params = {
        'epochs': 300,    # Prior posterior together
        'batch_size': 1,    # effectively x4
        'aggregation': 1,
        'learning_rate': 5e-4,
        'weight_decay': 2e-2,
        'num_classes': 14,
        'shape': (432, 224, 112),
        'alpha': (0.1, 2.0, 60), # JS Match of Prior and Likelihood
        'beta': (0.1, 2.0, 40), # Linear ramp up [min, max, epochs] VAE beta
        'compile': False,
        'autocast': True,
        'sw_batch_size': 1,
        'sw_overlap': 1/2,
        'data_augmentation': {
            # [I, Affine, Flip, Rotate90, Elastic]
            'spatial': [2, 2, 1, 1, 1],  
            # [I, Smooth, Noise, Bias, Contrast, Sharpen, Histogram]
            'intensity': [2, 2, 1, 0.5, 1, 1, 0.5],  
            # [I, Dropout, Shuffle]
            'coarse': [1, 1, 1]  
        }
    }
    output_dir = "VAEPosterior"
    comments = ["VAE Posterior pixdim = (1.5, 1.5, 2.5) - GTx2 + Aladdin training",
        "2 way prior decoding, No latent and z gradients",
        "No dropout stochastic depth in prior but yes posterior",
        f"beta-VAE only autoencodes labels, KL with normal and JS with each other",
        f"{train_params["shape"]} shape", 
        f"DiceFocal, 1-sample rand crop + 3-choose 1 augmentations [2, 1, 1]",
        f"Spatial {train_params['data_augmentation']['spatial']}; Intensity {train_params['data_augmentation']['intensity']}; Coarse {train_params['data_augmentation']['coarse']}"]
    
    gpu_count = torch.cuda.device_count()
    try:
        mp.spawn(
            main_worker,
            args=(gpu_count, model_params, train_params, output_dir, comments),
            nprocs=gpu_count,
            join=True)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main process. Terminating children...")
        mp.get_context('spawn')._shutdown()
