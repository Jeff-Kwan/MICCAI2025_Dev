import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # Fragmentation
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback
from datetime import datetime
from torch.optim import AdamW, lr_scheduler
from monai.data import DataLoader, Dataset
from monai.losses import DiceLoss, FocalLoss

from utils.dataset import get_transforms, get_data_files
from model.AttnUNet3 import AttnUNet3
from model.ViTSeg import ViTSeg
from model.ConvSeg import ConvSeg
from model.ConvSeg2 import ConvSeg2
from utils.ddp_trainer import DDPTrainer

torch.multiprocessing.set_sharing_strategy('file_system')

class SoftDiceFocalLoss(torch.nn.Module):
    def __init__(self, include_background=True, softmax=True, weight=None, lambda_focal=1.0, lambda_dice=1.0):
        super().__init__()
        self.dice_loss = DiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            softmax=softmax,
            weight=weight,
            soft_label=True)    # Use soft labels
        self.focal_loss = FocalLoss(
            include_background=include_background,
            to_onehot_y=False,
            use_softmax=softmax,
            weight=weight)
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_dice = self.dice_loss(inputs, targets)
        l_focal = self.focal_loss(inputs, targets)
        return self.lambda_dice * l_dice + self.lambda_focal * l_focal

def main_worker(rank: int,
                world_size: int,
                model,
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
        train_tf, val_tf = get_transforms(
            train_params['shape'],
            train_params['data_augmentation']['spatial'],
            train_params['data_augmentation']['intensity'],
            train_params['data_augmentation']['coarse'],
            soft=True)  # Use soft labels
        train_ds = Dataset(
            data=get_data_files(
                images_dir="data/nifti/train_gt/images",
                labels_dir="data/nifti/train_gt/softlabel",
                extension='.nii.gz') * 2
            + get_data_files(
                images_dir="data/nifti/train_pseudo/images",
                labels_dir="data/nifti/train_pseudo/softlabel",
                extension='.nii.gz'),
            transform=train_tf)
        val_ds = Dataset(
            data=get_data_files(
                images_dir="data/nifti/val/images",
                labels_dir="data/nifti/val/labels",
                extension='.nii.gz'),
            transform=val_tf)
        train_sampler = torch.utils.data.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = torch.utils.data.DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=train_params['batch_size'],
            sampler=train_sampler,
            num_workers=24,
            pin_memory=False,
            persistent_workers=True)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=False,
            persistent_workers=True)

        # Model, optimizer, scheduler, loss
        optimizer = AdamW(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['epochs'], eta_min=1e-6)
        criterion = SoftDiceFocalLoss(  # Use soft labels
            include_background=True, 
            softmax=True, 
            weight=torch.tensor([0.01] + train_params["weights"], device=rank),
            lambda_focal=1,
            lambda_dice=1)  


        # Initialize trainer and start
        trainer = DDPTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            train_params=train_params,
            output_dir=full_output,
            local_rank=rank,
            world_size=world_size,
            comments=comments)
        trainer.train(train_loader, val_loader, soft=True)

    except Exception as e:
        print(f"Rank {rank} crashed:", traceback.format_exc())
    finally:
        dist.destroy_process_group()


def get_comments(output_dir, train_params):
    return [
        f"{output_dir} - GT*2 (spatial soft) + pseudo (pred soft) labels - Loss modifier by dice error * 10 + 1",
        f"{train_params['shape']} shape, (2, 2, 1) patch embedding", 
        f"DiceFocal, 1-sample rand crop + augmentations -> also 0.2 label Gaussian smoothing",
        f"Spatial {train_params['data_augmentation']['spatial']}; Intensity {train_params['data_augmentation']['intensity']}; Coarse {train_params['data_augmentation']['coarse']}"
    ]


if __name__ == "__main__":
    # If needed:    pkill -f -- '--multiprocessing-fork'
    gpu_count = torch.cuda.device_count()
    architectures = ["AttnUNet3"]

    for architecture in architectures:
        model_params = json.load(open(f"configs/labellers/{architecture}/model.json"))
        train_params = json.load(open(f"configs/labellers/{architecture}/train.json"))
        output_dir = f"{architecture}"
        comments = get_comments(output_dir, train_params)

        print(f"Starting training for {architecture}...")
        if architecture == "AttnUNet3":
            model = AttnUNet3(model_params)
        elif architecture == "ConvSeg":
            model = ConvSeg(model_params)
        elif architecture == "ViTSeg":
            model = ViTSeg(model_params)
        elif architecture == "ConvSeg2":
            model = ConvSeg2(model_params)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        try:
            mp.spawn(
                main_worker,
                args=(gpu_count, model, train_params, output_dir, comments),
                nprocs=gpu_count,
                join=True)
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught in main process. Terminating children...")
            mp.get_context('spawn')._shutdown()
    
    
