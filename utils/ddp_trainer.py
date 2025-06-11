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

        # Device for this process (use local_rank directly)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        # Wrap in DDP if using multiple GPUs
        model.to(self.device)
        if self.world_size > 1:
            self.model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            self.model = model

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.precision = torch.bfloat16 if train_params.get("autocast", False) else torch.float32

        # Only rank 0 writes metrics
        self.dice_metric = mm.DiceMetric(include_background=False)
        if self.local_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            self.train_losses = []
            self.val_losses = []
            self.val_metrics = {'dice': []}
            self.best_results = {}
            self.model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.num_classes = train_params['num_classes']
            self.start_time = None

    def train(self, train_loader, val_loader):
        if self.local_rank == 0:
            self.start_time = time.time()

        epochs = self.train_params['epochs']
        agg_steps = self.train_params['aggregation']

        for epoch in range(epochs):
            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            self.model.train()
            running_loss = 0.0
            grad_norm = torch.tensor(0.0, device=self.device)

            loop = tqdm.tqdm(train_loader,
                             desc=f"[Rank {self.local_rank}] Epoch {epoch+1}/{epochs}",
                             disable=(self.local_rank!=0))
            self.optimizer.zero_grad()

            for i, batch in enumerate(loop):
                imgs = batch['image'].to(self.device, non_blocking=True)
                masks = batch['label'].to(self.device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=self.precision):
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, masks)
                loss.backward()
                running_loss += loss.item()

                if ((i + 1) % agg_steps == 0) or (i + 1 == len(train_loader)):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.local_rank == 0:
                    loop.set_postfix({'Norm': grad_norm.item(), 'Loss': loss.item()})

            self.scheduler.step()

            val_loss, metrics = self.evaluate(val_loader)
            if self.local_rank == 0:
                self.train_losses.append(running_loss / len(train_loader))
                self.val_losses.append(val_loss)
                self.val_metrics['dice'].append(metrics['dice'])
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {self.train_losses[-1]:.5f} | "
                      f"Val Loss: {val_loss:.5f} | "
                      f"Val Dice: {metrics['dice']:.5f}")
                self.plot_results()
                self.save_checkpoint(epoch, metrics)

        # wait for all ranks before exit
        if self.world_size > 1:
            dist.barrier()

    def evaluate(self, data_loader):
        self.model.eval()
        # local accumulators
        loss_sum = 0.0
        sample_count = 0
        # reset MONAI dice
        self.dice_metric.reset()

        with torch.inference_mode():
            # loop over your shard
            for batch in tqdm.tqdm(
                data_loader,
                desc=f"[Rank {self.local_rank}] Validation",
                disable=(self.local_rank != 0),
            ):
                imgs = batch['image'].to(self.device, non_blocking=True)
                masks = batch['label'].to(self.device, non_blocking=True)
                B = imgs.size(0)
                sample_count += B

                # sliding window inference as before
                with torch.autocast(device_type='cuda', dtype=self.precision):
                    aggregated = torch.zeros(
                        (B, self.num_classes, *imgs.shape[2:]),
                        device=self.device
                    )
                    for b in range(B):
                        logits = sliding_window_inference(
                            imgs[b:b+1],
                            roi_size=self.train_params['shape'],
                            sw_batch_size=self.train_params.get('sw_batch_size', 1),
                            predictor=lambda x: self.model(x),
                            overlap=self.train_params.get('sw_overlap', 0.25),
                            mode="gaussian",
                        )
                        aggregated[b] = logits

                    loss = self.criterion(aggregated, masks)
                # accumulate loss weighted by batch size
                loss_sum += loss.item() * B

                # one‐hot encode and update dice
                preds = one_hot(
                    torch.argmax(aggregated, dim=1, keepdim=True),
                    num_classes=self.num_classes
                )
                gts = one_hot(masks, num_classes=self.num_classes)
                self.dice_metric(y_pred=preds, y=gts)

        # after local loop, get local dice sum:
        # MONAI's DiceMetric with default reduction='mean_batch' gives mean per batch,
        # so to get a sum over all samples, multiply by sample_count:
        local_dice_mean = self.dice_metric.aggregate().item()
        local_dice_sum = local_dice_mean * sample_count
        self.dice_metric.reset()

        # convert to tensors
        loss_tensor = torch.tensor(loss_sum, device=self.device)
        samples_tensor = torch.tensor(sample_count, device=self.device)
        dice_tensor = torch.tensor(local_dice_sum, device=self.device)

        # all‐reduce across ranks
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(dice_tensor, op=dist.ReduceOp.SUM)

        # compute global means
        total_loss = loss_tensor.item() / samples_tensor.item()
        total_dice = dice_tensor.item() / samples_tensor.item()

        return total_loss, {'dice': total_dice}

    def save_checkpoint(self, epoch: int, val_metrics: dict):
        # Save last
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model.pth'))
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
