import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from datetime import datetime
from torch.optim import AdamW, lr_scheduler
from monai.data import PersistentDataset, DataLoader
from monai.losses import DiceCELoss
from monai.utils.enums import MetaKeys, SpaceKeys, TraceKeys
from monai.data.meta_tensor import MetaTensor

from utils import get_transforms, get_data_files
from model.Harmonics import HarmonicSeg
from utils.ddp_trainer import DDPTrainer

class SafeGlobalsContext:
    def __enter__(self):
        torch.serialization.add_safe_globals([
            np.dtype, np.ndarray, np.core.multiarray._reconstruct,
            np.dtypes.Int64DType, np.dtypes.Int32DType, np.dtypes.Int16DType,
            np.dtypes.UInt8DType, np.dtypes.Float32DType, np.dtypes.Float64DType,
            MetaKeys, SpaceKeys, TraceKeys, MetaTensor
        ])

    def __exit__(self, exc_type, exc_value, traceback):
        pass

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
        train_tf, val_tf = get_transforms(train_params['shape'], train_params['num_crops'])
        with SafeGlobalsContext():
            train_ds = PersistentDataset(
                # data=get_data_files(
                #     images_dir="data/preprocessed/train_gt/images",
                #     labels_dir="data/preprocessed/train_gt/labels"),
                data=get_data_files(
                    images_dir="data/preprocessed/train_pseudo/images",
                    labels_dir="data/preprocessed/train_pseudo/aladdin5"),
                transform=train_tf,
                cache_dir="data/cache/pseudo_label")
            train_sampler = torch.utils.data.DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            train_loader = DataLoader(
                train_ds,
                batch_size=train_params['batch_size'],
                sampler=train_sampler,
                num_workers=30,
                prefetch_factor=1,
                pin_memory=True,
                persistent_workers=True)
            if rank == 0:
                val_ds = PersistentDataset(
                    data=get_data_files(
                        images_dir="data/preprocessed/val/images",
                        labels_dir="data/preprocessed/val/labels"),
                    transform=val_tf,
                    cache_dir="data/cache/val")
                val_loader = DataLoader(
                    val_ds,
                    batch_size=1,
                    shuffle=False,
                    num_workers=24,
                    persistent_workers=False)
            else:
                val_loader = None

            # Model, optimizer, scheduler, loss
            model = HarmonicSeg(model_params)
            optimizer = AdamW(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['epochs'])
            criterion = DiceCELoss(
                include_background=False, 
                to_onehot_y=True, 
                softmax=True, 
                weight=torch.tensor([0.01] + [1.0] * 13, device=rank),
                label_smoothing=0.1,
                lambda_ce=0.34,
                lambda_dice=0.66,)

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
            trainer.train(train_loader, val_loader)

            # Final evaluations on rank 0
            if rank == 0:
                test_loss, test_metrics = trainer.evaluate(val_loader)
                with open(os.path.join(full_output, 'results.txt'), 'a') as f:
                    f.write(f"\nLast Model Test: Loss={test_loss:.5f}, Dice={test_metrics['dice']:.5f}\n")
                print(f"[Last] Test Loss: {test_loss:.5f}, Dice: {test_metrics['dice']:.5f}")

                # Load best and re-evaluate
                trainer.model.load_state_dict(torch.load(os.path.join(full_output, 'best_model.pth')))
                best_loss, best_metrics = trainer.evaluate(val_loader)
                with open(os.path.join(full_output, 'results.txt'), 'a') as f:
                    f.write(f"Best Model Test: Loss={best_loss:.5f}, Dice={best_metrics['dice']:.5f}\n")
                print(f"[Best] Test Loss: {best_loss:.5f}, Dice: {best_metrics['dice']:.5f}")

    except KeyboardInterrupt:
        print(f"Rank {rank}: Received KeyboardInterrupt, cleaning up...")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # If needed:    pkill -f -- '--multiprocessing-fork'
    # Load configs
    model_params = json.load(open("configs/model/large.json"))
    train_params = {
        'epochs': 200,
        'batch_size': 1,    # effectively x4
        'aggregation': 1,
        'learning_rate': 1e-3,
        'weight_decay': 2e-2,
        'num_classes': 14,
        'shape': (160, 160, 80),
        'num_crops': 8,
        'compile': True,
        'autocast': True,
        'sw_batch_size': 16,
        'sw_overlap': 1/8
    }
    output_dir = "PseudolabelsAll"
    comments = ["HarmonicSeg Large - 2000 Aladdin5 training",
        "(160, 160, 80) shape", 
        "DiceCE, 8-sample rand crop + fewer augmentations",
        "Spatial [2, 2, 0, 0, 1]; Intensity [3, 2, 1, 0, 1, 0, 0]; Coarse [3, 1, 1]"]
    torch._dynamo.config.cache_size_limit = 32  # Up the cache size limit for dynamo

    try:
        mp.spawn(
            main_worker,
            args=(4, model_params, train_params, output_dir, comments),
            nprocs=4,
            join=True)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main process. Terminating children...")
        mp.get_context('spawn')._shutdown()
