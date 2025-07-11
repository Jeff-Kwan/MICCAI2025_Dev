import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # Fragmentation
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback
from datetime import datetime
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset
from monai.data import ThreadDataLoader, Dataset
from monai.losses import DiceLoss, FocalLoss

from utils.dataset import get_dual_transforms, get_dual_data_files, get_data_files
from model.AttnUNet import AttnUNet
from model.ConvSeg import ConvSeg
from model.ConvSeg2 import ConvSeg2
from utils.dual_trainer import DDPTrainer

torch.multiprocessing.set_sharing_strategy('file_system')

class SoftDiceFocalLoss(torch.nn.Module):
    def __init__(self, include_background=True, softmax=True, weight=None, 
                 lambda_focal=1.0, lambda_dice=1.0, gamma=2.0):
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
            gamma=gamma,
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
                model1,
                model2,
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
        train_tf_gt, val_tf = get_dual_transforms(
            train_params['shape'],
            train_params['data_augmentation']['spatial'],
            train_params['data_augmentation']['intensity'],
            train_params['data_augmentation']['coarse'],
            gt=True)
        train_tf_pseudo, _ = get_dual_transforms(
            train_params['shape'],
            train_params['data_augmentation']['spatial'],
            train_params['data_augmentation']['intensity'],
            train_params['data_augmentation']['coarse'],
            gt=False)
        train_ds_gt = Dataset(
            data=get_dual_data_files(
                images_dir="data/nifti/train_gt/images",
                labels1_dir="data/nifti/train_gt/labels",
                labels2_dir="data/nifti/train_gt/labels",   # GT are the same
                extension='.nii.gz') * 4,
            transform=train_tf_gt)
        train_ds_pseudo = Dataset(
            data=get_dual_data_files(
                images_dir="data/nifti/train_pseudo/images",
                labels1_dir="data/nifti/train_pseudo/aladdin5",
                labels2_dir="data/nifti/train_pseudo/blackbean",
                extension='.nii.gz'),
            transform=train_tf_pseudo)
        train_ds = ConcatDataset([train_ds_gt, train_ds_pseudo])
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
        train_loader = ThreadDataLoader(
            train_ds,
            batch_size=train_params['batch_size'],
            sampler=train_sampler,
            num_workers=46,
            pin_memory=False,
            persistent_workers=True)
        val_loader = ThreadDataLoader(
            val_ds,
            batch_size=1,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=False,
            persistent_workers=True)

        # Model, optimizer, scheduler, loss
        # Cosine Annealing Warm Restarts on Half
        optimizer1 = AdamW(model1.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
        scheduler1 = CosineAnnealingLR(optimizer1, T_max=train_params['epochs'], eta_min=train_params['min_lr'])
        optimizer2 = AdamW(model2.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
        scheduler2 = CosineAnnealingLR(optimizer2, T_max=train_params['epochs'], eta_min=train_params['min_lr'])
        criterion = SoftDiceFocalLoss(  # Use soft labels
            include_background=True, 
            softmax=True, 
            weight=torch.tensor([0.01] + train_params["weights"], device=rank),
            lambda_focal=1,
            lambda_dice=1)


        # Initialize trainer and start
        trainer = DDPTrainer(
            model1=model1,
            model2=model2,
            optimizer1=optimizer1,
            optimizer2=optimizer2,
            criterion=criterion,
            scheduler1=scheduler1,
            scheduler2=scheduler2,
            train_params=train_params,
            output_dir=full_output,
            local_rank=rank,
            world_size=world_size,
            comments=comments)
        trainer.train(train_loader, val_loader)

    except Exception as e:
        print(f"Rank {rank} crashed:", traceback.format_exc())
    finally:
        dist.destroy_process_group()


def get_comments(output_dir, train_params):
    return [
        f"{output_dir} - GT*4 (hard) + pseudo (pred soft) labels",
        "ConvSeg2 + ConvSeg2, dual model with stochastic cross or sum teacher",
        f"{train_params['shape']} shape, (2, 2, 1) patch embedding", 
        f"SoftDiceFocal, 1-sample rand crop + augmentations",
        f"Spatial {train_params['data_augmentation']['spatial']}; Intensity {train_params['data_augmentation']['intensity']}; Coarse {train_params['data_augmentation']['coarse']}"
    ]


if __name__ == "__main__":
    # If needed:    pkill -f -- '--multiprocessing-fork'
    # Must train with batch size 1 because MONAI centercrop label
    gpu_count = torch.cuda.device_count()
    model1_params = json.load(open(f"configs/dual/convseg2.json"))
    model2_params = json.load(open(f"configs/dual/convseg2.json"))
    train_params = json.load(open(f"configs/dual/train.json"))

    output_dir = f"DualModel"
    comments = get_comments(output_dir, train_params)


    print(f"Starting training for Dual Models...")
    # model1 = AttnUNet(model1_params)
    model1 = ConvSeg2(model2_params)
    model2 = ConvSeg2(model2_params)
    
    try:
        mp.spawn(
            main_worker,
            args=(gpu_count, model1, model2, train_params, output_dir, comments),
            nprocs=gpu_count,
            join=True)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main process. Terminating children...")
        mp.get_context('spawn')._shutdown()

    
