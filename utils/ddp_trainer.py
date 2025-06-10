# ddp_trainer.py

import os
import json
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm
import matplotlib.pyplot as plt
import monai.metrics as mm
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference

def setup_ddp(local_rank: int, world_size: int):
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        dist.init_process_group(backend="nccl",
                                rank=local_rank,
                                world_size=world_size)
    torch.cuda.set_device(local_rank)

class DDPTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scheduler,
        train_params: dict,
        output_dir: str,
        local_rank: int = 0,
        world_size: int = 1,
        comments: list = None,
    ):
        self.local_rank = local_rank
        self.world_size = world_size
        self.comments = comments or []
        self.train_params = train_params
        self.output_dir = output_dir

        # Device for this process
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        # Initialize DDP if needed
        if world_size > 1:
            setup_ddp(local_rank, world_size)
            model.to(self.device)
            self.model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            self.model = model.to(self.device)

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.precision = torch.bfloat16 if train_params.get("autocast", False) else torch.float32

        # Only rank 0 tracks and writes metrics
        if self.local_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            self.train_losses = []
            self.val_losses = []
            self.val_metrics = {'dice': []}
            self.best_results = {}
            self.model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.num_classes = train_params['num_classes']
            self.dice_metric = mm.DiceMetric(include_background=False)
            self.start_time = None

    def train(self, train_loader, val_loader):
        rank = dist.get_rank() if self.world_size > 1 else 0
        if rank == 0:
            self.start_time = time.time()

        epochs = self.train_params['epochs']
        agg_steps = self.train_params['aggregation']

        for epoch in range(epochs):
            # Shuffle sampler if distributed
            if self.world_size > 1 and hasattr(train_loader, 'sampler'):
                train_loader.sampler.set_epoch(epoch)

            self.model.train()
            running_loss = 0.0
            grad_norm = torch.tensor(0.0, device=self.device)

            loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=(rank!=0))
            self.optimizer.zero_grad()

            for i, batch in enumerate(loop):
                imgs = batch['image'].to(self.device, non_blocking=True)
                masks = batch['label'].to(self.device, non_blocking=True)

                with torch.autocast('cuda', dtype=self.precision):
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, masks)
                loss.backward()
                running_loss += loss.item()

                if ((i + 1) % agg_steps == 0) or (i + 1 == len(train_loader)):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if rank == 0:
                    loop.set_postfix({'Norm': grad_norm.item(), 'Loss': loss.item()})

            self.scheduler.step()

            # Validation (only metrics computed on rank 0)
            if rank == 0:
                val_loss, metrics = self.evaluate(val_loader)
                self.train_losses.append(running_loss / len(train_loader))
                self.val_losses.append(val_loss)
                self.val_metrics['dice'].append(metrics['dice'])

                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {self.train_losses[-1]:.5f} | "
                    f"Val Loss: {val_loss:.5f} | "
                    f"Val Dice: {metrics['dice']:.5f}"
                )

                self.plot_results()
                self.save_checkpoint(epoch, metrics)

        # Clean up distributed state
        if self.world_size > 1:
            dist.barrier()

    def evaluate(self, data_loader):
        self.model.eval()
        loss_sum = 0.0
        self.dice_metric.reset()

        with torch.inference_mode():
            loop = tqdm.tqdm(data_loader, desc='Validation', disable=(dist.get_rank()!=0))
            for batch in loop:
                imgs = batch['image'].to(self.device, non_blocking=True)
                masks = batch['label'].to(self.device, non_blocking=True)
                B, C, H, W, D = imgs.shape

                with torch.autocast('cuda', dtype=self.precision):
                    # Sliding window per-volume inference
                    aggregated = torch.zeros((B, self.num_classes, H, W, D), device=self.device)
                    for b in range(B):
                        single = imgs[b:b+1]
                        logits = sliding_window_inference(
                            inputs=single,
                            roi_size=self.train_params['shape'],
                            sw_batch_size=self.train_params.get('sw_batch_size', 1),
                            predictor=lambda x: self.model(x),
                            overlap=self.train_params.get('sw_overlap', 0.25),
                            mode="gaussian"
                        )
                        aggregated[b] = logits

                    loss = self.criterion(aggregated, masks)
                loss_sum += loss.item()

                preds = one_hot(torch.argmax(aggregated, dim=1, keepdim=True),
                                num_classes=self.num_classes)
                gts   = one_hot(masks, num_classes=self.num_classes)
                self.dice_metric(y_pred=preds, y=gts)

        mean_loss = loss_sum / len(data_loader)
        dice_score = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        return mean_loss, {'dice': dice_score}

    def save_checkpoint(self, epoch: int, val_metrics: dict):
        # Save last
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model.pth'))
        # Save metrics history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(history, f, indent=4)

        # Save best
        if self.val_metrics['dice'][-1] >= max(self.val_metrics['dice'][:-1] + [0]):
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
            self.best_results = {
                'epoch': epoch,
                'train_loss': self.train_losses[-1],
                'val_loss': self.val_losses[-1],
                'val_metrics': val_metrics
            }

        # Write summary
        elapsed = time.time() - self.start_time
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        with open(os.path.join(self.output_dir, 'results.txt'), 'w') as f:
            f.write(f"Model size: {self.model_size/1e6:.2f}M\n")
            f.write(f"Training time: {int(hrs):02}:{int(mins):02}:{int(secs):02}\n\n")
            for c in self.comments:
                f.write(c + "\n")
            f.write(f"\nTrain params: {json.dumps(self.train_params, indent=4)}\n")
            f.write(f"\nBest results: {json.dumps(self.best_results, indent=4)}\n")

    def plot_results(self):
        epochs = range(1, len(self.train_losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss curve
        ax1.plot(epochs, self.train_losses, label='Train')
        ax1.plot(epochs, self.val_losses, label='Val')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.legend(); ax1.set_title('Loss')

        # Dice curve
        ax2.plot(epochs, self.val_metrics['dice'], label='Val Dice')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice')
        ax2.legend(); ax2.set_title('Validation Dice')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close(fig)
