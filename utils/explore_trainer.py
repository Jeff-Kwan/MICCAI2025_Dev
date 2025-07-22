import os
import json
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import monai.metrics as mm
from monai.transforms import CenterSpatialCrop, Rand3DElastic
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn

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
            self.model = DDP(model, device_ids=[self.local_rank], 
                output_device=self.local_rank, broadcast_buffers=False)

        else:
            self.model = model
        self.ema_model = AveragedModel(self.model, self.device,
                            get_ema_avg_fn(train_params["ema_decay"]), 
                            use_buffers=False)

        # Pseudo label mix
        alpha = train_params["alpha"]
        lower, upper = train_params["alpha_range"] # Avoid all zero labels
        self.alpha = torch.cat([    
            torch.zeros(alpha[0], device=self.device),
            torch.linspace(lower, upper, alpha[1] - alpha[0], device=self.device),
            torch.ones(train_params['epochs'] - alpha[1], device=self.device) * upper
        ])
        self.alpha_mix = train_params["alpha_mix"]
        self.epsilon = torch.linspace(train_params["epsilon_range"][0],
                                      train_params["epsilon_range"][1], 
                                      train_params['epochs'], device=self.device)
        self.pred_threshold = train_params.get("pred_threshold", 1/3)
        self.sharpen = train_params.get("sharpen", 2.0)
        self.elastic = Rand3DElastic(
                prob=1.0,
                sigma_range=(1.5, 2.0),
                magnitude_range=(2, 12),
                # No affine or large spatial mismatch!
                mode="nearest")
        self.center_crop = CenterSpatialCrop(train_params['shape'])

        # Optimizations
        if train_params.get('autocast', False):
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')
        if train_params.get("compile", False):
            self.model = torch.compile(self.model)

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.precision = torch.bfloat16 if train_params.get("autocast", False) else torch.float32

        # Only rank 0 writes metrics
        self.num_classes = train_params['num_classes']
        self.dice_metric = mm.DiceMetric(include_background=True, 
                                         ignore_empty=False,
                                         reduction='mean_batch')
        if self.local_rank == 0:
            self.train_losses = []
            self.val_losses = []
            self.val_metrics = {'dice': [], 'class_dice': []}
            self.model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.start_time = None
            self.class_names = ["Liver", "Right kidney", "Spleen", "Pancreas", 
                                "Aorta", "Inferior Vena Cava", "Right Adrenal Gland", "Left Adrenal Gland",
                                "Gallbladder", "Esophagus", "Stomach", "Duodenum", "Left kidney"]

    def train(self, train_loader, val_loader=None):
        if self.local_rank == 0:
            self.start_time = time.time()

        epochs = self.train_params['epochs']
        agg_steps = self.train_params['aggregation']
        iters = len(train_loader)

        for epoch in range(epochs):
            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            self.model.train()
            self.ema_model.update_parameters(self.model.module)
            running_loss = 0.0
            grad_norm = torch.tensor(0.0, device=self.device)

            loop = tqdm.tqdm(train_loader,
                             desc=f"[Rank {self.local_rank}] Epoch {epoch+1}/{epochs}",
                             disable=(self.local_rank!=0))
            self.optimizer.zero_grad()

            for i, batch in enumerate(loop):
                img = batch['image'].to(self.device, non_blocking=True)
                clean_img = batch['clean_image'].to(self.device, non_blocking=True)
                label = batch['label'].to(self.device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=self.precision):
                    logits = self.model(img)

                    with torch.no_grad():
                        label = label.float() / 255.0  # Convert to float on GPU

                        if batch['gt'] == False:
                            # Swap in EMA
                            if torch.rand(1).item() < self.alpha[epoch]:
                                self.ema_model.eval()
                                pred = sliding_window_inference(
                                            clean_img,
                                            roi_size=self.train_params['shape'],
                                            sw_batch_size=8,
                                            predictor=lambda x: self.ema_model(x),
                                            overlap=0.5,
                                            mode="gaussian")

                                # EMA teacher
                                pred = torch.softmax(pred * self.sharpen, dim=1)
                                pred = pred * (pred > self.pred_threshold)
                                label = self.alpha_mix * pred + (1-self.alpha_mix) * label
                                label = F.normalize(label, p=1, dim=1)

                        # Center crop to original shape
                        label = self.center_crop(label.squeeze(0)).unsqueeze(0)

                        # Exploration ie. Label Distortion (cannot autocast elastic)
                        with torch.autocast('cuda', enabled=False):
                            if torch.rand(1).item() < self.epsilon[epoch]:  
                                label = self.elastic(label.squeeze(0)).unsqueeze(0)

                    loss = self.criterion(logits, label)

                loss.backward()
                running_loss += loss.item()

                if ((i + 1) % agg_steps == 0) or (i + 1 == iters):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.ema_model.update_parameters(self.model.module)

                if self.local_rank == 0:
                    loop.set_postfix({'Norm': f'{grad_norm.item():.3f}', 'Loss': f'{loss.item():.3f}'})

            # Step schedulers
            self.scheduler.step()

            val_loss, metrics = self.evaluate(self.model, val_loader)
            if self.world_size > 1:
                torch.cuda.synchronize(self.device)
                dist.barrier()
            if self.local_rank == 0 and val_loader is not None:
                metrics["dice"] = float(sum(metrics["class_dice"][1:]) / len(metrics["class_dice"][1:]))
                self.train_losses.append(running_loss / len(train_loader))
                self.val_losses.append(val_loss)
                self.val_metrics['dice'].append(metrics['dice'])
                self.val_metrics['class_dice'].append(metrics['class_dice'])
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {self.train_losses[-1]:.5f} | "
                      f"Val Loss: {val_loss:.5f} | "
                      f"Val Dice: {metrics['dice']:.5f}")
                self.plot_results()
                self.save_checkpoint(epoch)


    @torch.no_grad()
    def evaluate(self, model, data_loader):
        model.eval()
        # local accumulators
        loss_sum = torch.tensor(0.0, device=self.device)
        sample_count = torch.tensor(0, device=self.device)
        # reset MONAI dice
        self.dice_metric.reset()

        # loop over your shard
        for batch in tqdm.tqdm(
            data_loader,
            desc=f"[Rank {self.local_rank}] Validation",
            disable=(self.local_rank != 0),
        ):
            imgs = batch['image'].to(self.device, non_blocking=True)
            masks = batch['label'].to(self.device, non_blocking=True)
            masks = one_hot(masks, num_classes=self.num_classes)
            B = imgs.size(0)
            sample_count += B

            # sliding window inference as before
            with torch.autocast(device_type='cuda', dtype=self.precision):
                aggregated = sliding_window_inference(
                    imgs,
                    roi_size=self.train_params['shape'],
                    sw_batch_size=self.train_params.get('sw_batch_size', 1),
                    predictor=lambda x: model(x),
                    overlap=self.train_params.get('sw_overlap', 0.25),
                    mode="gaussian",
                    buffer_steps=None
                )

                loss = self.criterion(aggregated, masks)
            # accumulate loss weighted by batch size
            loss_sum += loss.item() * B

            # one‐hot encode and update dice
            preds = one_hot(
                torch.argmax(aggregated, dim=1, keepdim=True),
                num_classes=self.num_classes)
            self.dice_metric(y_pred=preds, y=masks)

        # Aggregate loss and sample count across all ranks
        if self.world_size > 1:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)

        total_loss = loss_sum.item() / max(sample_count.item(), 1)
        total_dice = list(self.dice_metric.aggregate().cpu().numpy())
        total_dice = [float(d) for d in total_dice]  # Convert to float for JSON serialization
        return total_loss, {'class_dice': total_dice}

    def save_checkpoint(self, epoch: int):
        # Save last
        state_dict = (self.model.module.state_dict()
                if isinstance(self.model, DDP) else self.model.state_dict())
        torch.save(state_dict, os.path.join(self.output_dir, 'model.pth'))
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(history, f, indent=4)

        # Save EMA model
        ema_state_dict = (self.ema_model.module.state_dict()
                if isinstance(self.ema_model, DDP) else self.ema_model.state_dict())
        torch.save(ema_state_dict, os.path.join(self.output_dir, 'ema_model.pth'))

        # Write summary
        elapsed = time.time() - self.start_time
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        with open(os.path.join(self.output_dir, 'results.txt'), 'w') as f:
            f.write(f"Model size: {self.model_size/1e6:.2f}M\n")
            f.write(f"Training time: {int(hrs):02}:{int(mins):02}:{int(secs):02}\n\n")
            f.write(f"Epoch {epoch+1} results:\n")
            f.write(f"Train Loss: {self.train_losses[-1]:.5f}; Val Loss: {self.val_losses[-1]:.5f}; Val Dice: {self.val_metrics['dice'][-1]:.5f}\n\n")
            for c in self.comments:
                f.write(c + "\n")
            f.write(f"\nModel params: {json.dumps(self.model.module.model_params, indent=4)}\n")
            f.write(f"\nTrain params: {json.dumps(self.train_params, indent=4)}\n")

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

        # Plot class dice
        plt.figure(figsize=(12, 6))
        class_dice = np.array(self.val_metrics["class_dice"]).transpose()[1:].tolist()
        for name, dice in zip(self.class_names, class_dice):
            plt.plot(dice, label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.title("Dice Score for Each Organ over Training")
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'class_dice.png'))
        plt.close()

