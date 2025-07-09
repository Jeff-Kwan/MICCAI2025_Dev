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
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

class DDPTrainer:
    def __init__(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        optimizer1: torch.optim.Optimizer,
        optimizer2: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scheduler1,
        scheduler2,
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
        model1.to(self.device); model2.to(self.device)
        if self.world_size > 1:
            self.model1 = DDP(model1, device_ids=[self.local_rank], 
                output_device=self.local_rank, broadcast_buffers=False)
            self.model2 = DDP(model2, device_ids=[self.local_rank], 
                output_device=self.local_rank, broadcast_buffers=False)
        else:
            self.model1 = model1
            self.model2 = model2

        # Pseudo label mix
        alpha = train_params["alpha"]
        self.alpha = torch.cat([    # 0.999 to avoid all zero labels
            torch.zeros(alpha[0], device=self.device),
            torch.linspace(0, 0.999, alpha[1] - alpha[0], device=self.device),
            torch.ones(train_params['epochs'] - alpha[1], device=self.device) * 0.999
        ])

        # Optimizations
        if train_params.get('autocast', False):
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('medium')
        if train_params.get("compile", False):
            self.model1 = torch.compile(self.model1)
            self.model2 = torch.compile(self.model2)

        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.criterion = criterion
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.precision = torch.bfloat16 if train_params.get("autocast", False) else torch.float32

        # Only rank 0 writes metrics
        self.num_classes = train_params['num_classes']
        self.dice_metric = mm.DiceMetric(include_background=True, 
                                         ignore_empty=False,
                                         reduction='mean_batch')
        if self.local_rank == 0:
            self.train_losses = {"model1": [],
                                "model2": []}
            self.val_losses = {"model1": [],
                                "model2": []}
            self.val_metrics = {"model1": {'dice': [], 'class_dice': []},
                                "model2": {'dice': [], 'class_dice': []}}
            self.model_size1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
            self.model_size2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
            self.start_time = None
            self.class_names = ["Liver", "Right kidney", "Spleen", "Pancreas", 
                                "Aorta", "Inferior Vena Cava", "Right Adrenal Gland", 
                                "Gallbladder", "Esophagus", "Stomach", "Duodenum", "Left kidney"]

    def train(self, train_loader, val_loader=None):
        if self.local_rank == 0:
            self.start_time = time.time()

        epochs = self.train_params['epochs']
        agg_steps = self.train_params['aggregation']

        for epoch in range(epochs):
            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            self.model1.train(); self.model2.train()
            running_loss1 = 0.0; running_loss2 = 0.0
            grad_norm1 = torch.tensor(0.0, device=self.device)
            grad_norm2 = torch.tensor(0.0, device=self.device)

            loop = tqdm.tqdm(train_loader,
                             desc=f"[Rank {self.local_rank}] Epoch {epoch+1}/{epochs}",
                             disable=(self.local_rank!=0))
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()

            for i, batch in enumerate(loop):
                img = batch['image'].to(self.device, non_blocking=True)
                clean_img = batch['clean_image'].to(self.device, non_blocking=True)
                label1, label2 = ((batch['label1'], batch['label2']) 
                                  if torch.rand(1).item() < 0.5 # Flip between labels
                                  else (batch['label2'], batch['label1']))
                label1 = label1.to(self.device, non_blocking=True)
                label2 = label2.to(self.device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=self.precision):
                    logits1 = self.model1(img)
                    logits2 = self.model2(clean_img)

                    # No division ie. x2; T=0.5
                    logits_total = logits1.detach().clone() + logits2.detach().clone()    
                    pred_total = torch.softmax(logits_total, dim=1)
                    pred_total = pred_total * (pred_total >= 0.25)  # At most 3 preds

                    label1_pseudo = self.alpha[epoch] * pred_total + (1-self.alpha[epoch]) * one_hot(label1, self.num_classes)
                    label1_pseudo = F.normalize(label1_pseudo, p=1, dim=1)

                    label2_pseudo = self.alpha[epoch] * pred_total + (1-self.alpha[epoch]) * one_hot(label2, self.num_classes)
                    label2_pseudo = F.normalize(label2_pseudo, p=1, dim=1)

                    loss1 = self.criterion(logits1, label1_pseudo)
                    loss2 = self.criterion(logits2, label2_pseudo)

                loss1.backward()
                loss2.backward()
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()

                if ((i + 1) % agg_steps == 0) or (i + 1 == len(train_loader)):
                    grad_norm1 = torch.nn.utils.clip_grad_norm_(self.model1.parameters(), 1.0)
                    grad_norm2 = torch.nn.utils.clip_grad_norm_(self.model2.parameters(), 1.0)
                    self.optimizer1.step()
                    self.optimizer1.zero_grad()
                    self.optimizer2.step()
                    self.optimizer2.zero_grad()

                if self.local_rank == 0:
                    loop.set_postfix({'Norm': (grad_norm1.item(), grad_norm2.item()), 'Loss': (loss1.item(), loss2.item())})

            self.scheduler1.step()
            self.scheduler2.step()

            val_loss1, metrics1 = self.evaluate(self.model1, val_loader)
            val_loss2, metrics2 = self.evaluate(self.model2, val_loader)
            if self.world_size > 1:
                torch.cuda.synchronize(self.device)
                dist.barrier()
            if self.local_rank == 0 and val_loader is not None:
                metrics1["dice"] = float(sum(metrics1["class_dice"]) / len(metrics1["class_dice"]))
                metrics2["dice"] = float(sum(metrics2["class_dice"]) / len(metrics2["class_dice"]))
                self.train_losses["model1"].append(running_loss1 / len(train_loader))
                self.train_losses["model2"].append(running_loss2 / len(train_loader))
                self.val_losses["model1"].append(val_loss1)
                self.val_losses["model2"].append(val_loss2)
                self.val_metrics["model1"]['dice'].append(metrics1['dice'])
                self.val_metrics["model2"]['dice'].append(metrics2['dice'])
                self.val_metrics["model1"]['class_dice'].append(metrics1['class_dice'])
                self.val_metrics["model2"]['class_dice'].append(metrics2['class_dice'])
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {running_loss1 / len(train_loader):.4f}, {running_loss2 / len(train_loader):.4f} | "
                      f"Val Loss: {val_loss1:.4f}, {val_loss2:.4f} | "
                      f"Val Dice: {metrics1['dice']:.4f}, {metrics1['dice']:.4f}")
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
        state_dict = (self.model1.module.state_dict()
                if isinstance(self.model, DDP) else self.model.state_dict())
        torch.save(state_dict, os.path.join(self.output_dir, 'model1.pth'))
        state_dict = (self.model2.module.state_dict()
                if isinstance(self.model, DDP) else self.model.state_dict())
        torch.save(state_dict, os.path.join(self.output_dir, 'model2.pth'))
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(history, f, indent=4)

        # Write summary
        elapsed = time.time() - self.start_time
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        with open(os.path.join(self.output_dir, 'results.txt'), 'w') as f:
            f.write(f"Model size: {self.model_size1/1e6:.2f}M, {self.model_size2/1e6:.2f}M\n")
            f.write(f"Training time: {int(hrs):02}:{int(mins):02}:{int(secs):02}\n\n")
            f.write(f"Epoch {epoch+1} results:\n")
            f.write(f"Train Loss: {self.train_losses["model1"][-1]:.5f}, {self.train_losses["model2"][-1]:.5f}; \
                    Val Loss: {self.val_losses["model2"][-1]:.5f}, {self.val_losses['model2'][-1]:.5f}; \
                    Val Dice: {self.val_metrics["model1"]['dice'][-1]:.5f}, {self.val_metrics["model2"]['dice'][-1]:.5f}\n\n")
            for c in self.comments:
                f.write(c + "\n")
            f.write(f"\nModel1 params: {json.dumps(self.model1.module.model_params, indent=4)}\n")
            f.write(f"\nModel2 params: {json.dumps(self.model2.module.model_params, indent=4)}\n")
            f.write(f"\nTrain params: {json.dumps(self.train_params, indent=4)}\n")

    def plot_results(self):
        epochs = range(1, len(self.train_losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss curve
        ax1.plot(epochs, self.train_losses["model1"], label='Train Model1', color='blue')
        ax1.plot(epochs, self.train_losses["model2"], label='Train Model2', color='green')
        ax1.plot(epochs, self.val_losses["model1"], label='Val Model1', linestyle='--', color='blue')
        ax1.plot(epochs, self.val_losses["model2"], label='Val Model2', linestyle='--', color='green')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.legend(); ax1.set_title('Loss')

        # Dice curve
        ax2.plot(epochs, self.val_metrics["model1"]['dice'], label='Val Dice Model1', color='blue')
        ax2.plot(epochs, self.val_metrics["model2"]['dice'], label='Val Dice Model2', color='green')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice')
        ax2.legend(); ax2.set_title('Validation Dice')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close(fig)

        # Plot class dice
        plt.figure(figsize=(12, 6))
        class_dice = np.array(self.val_metrics["model1"]["class_dice"]).transpose()[1:].tolist()
        for name, dice in zip(self.class_names, class_dice):
            plt.plot(dice, label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.title("Model1 Dice Score for Each Organ over Training")
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'class_dice_m1.png'))
        plt.close()

        plt.figure(figsize=(12, 6))
        class_dice = np.array(self.val_metrics["model2"]["class_dice"]).transpose()[1:].tolist()
        for name, dice in zip(self.class_names, class_dice):
            plt.plot(dice, label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.title("Model2 Dice Score for Each Organ over Training")
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'class_dice_m2.png'))
        plt.close()
