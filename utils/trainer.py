import torch
import tqdm
import os
import json
import matplotlib.pyplot as plt
from time import time

import monai.metrics as mm
from monai.metrics.surface_distance import SurfaceDistanceMetric
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference

class Trainer():
    def __init__(self, model, optimizer, criterion, scheduler, train_params, output_dir, device, comments):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_params = train_params
        self.output_dir = output_dir
        self.device = device
        self.comments = comments

        self.train_losses = []
        self.val_losses = []
        self.val_metrics = {
            'dice': [],
            'surf_dist': []
        }
        self.best_results = {}

        os.makedirs(output_dir, exist_ok=True)
        self.model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Number of segmentation classes (including background)
        self.num_classes = self.train_params['num_classes']

        # MONAI Dice metric: average over batch and classes, ignore background (channel=0)
        self.dice_metric = mm.DiceMetric(
            include_background=False,
        )

        # MONAI Surface Distance metric (mean symmetric distance):
        #  - include_background=False → ignore channel 0
        #  - reduction=MEAN_BATCH → average over classes & batch
        #  - distance_metric="euclidean" → Euclidean distance in voxel units
        #  - symmetric=True → average pred→GT and GT→pred distances
        self.surface_dist_metric = SurfaceDistanceMetric(
            include_background=False,
            # symmetric=True
        )

        self.start_time = None

    def train(self, train_loader, val_loader):
        self.model.to(self.device)
        epochs = self.train_params['epochs']
        aggregation = self.train_params['aggregation']

        self.start_time = time()
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            norm = torch.tensor(0.0).to(self.device)
            p_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            self.optimizer.zero_grad()

            for i, dict in enumerate(p_bar):
                imgs = dict["image"].to(self.device, non_blocking=True)          # [B, C_img, H, W, D]
                masks = dict["label"].to(self.device, non_blocking=True)        # [B, H, W, D] (integer labels)

                outputs = self.model(imgs)                               # logits [B, num_classes, H, W, D]
                loss = self.criterion(outputs, masks)                    # existing loss function
                loss.backward()
                train_loss += loss.item()

                # Gradient accumulation
                if ((i+1) % aggregation == 0) or (i == len(train_loader)-1):
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                p_bar.set_postfix({'Norm': norm.item(), 'Loss': loss.item()})

            self.scheduler.step()

            # Evaluate on validation set
            val_loss, metrics = self.evaluate(val_loader)
            self.train_losses.append(train_loss / len(train_loader))
            self.val_losses.append(val_loss)
            self.val_metrics['dice'].append(metrics['dice'])
            self.val_metrics['surf_dist'].append(metrics['surf_dist'])

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {self.train_losses[-1]:.5f} | "
                f"Val Loss: {self.val_losses[-1]:.5f} | "
                f"Val Dice: {metrics['dice']:.5f} | "
                f"Val SurfDist: {metrics['surf_dist']:.5f}"
            )

            # Save results, checkpoint, etc.
            self.plot_results()
            self.save_checkpoint(epoch, metrics)

    def evaluate(self, data_loader):
        """
        Runs one full pass over data_loader, computes the mean validation loss,
        and then uses MONAI's DiceMetric and SurfaceDistanceMetric to compute
        multi-class Dice and mean surface-distance scores over all batches.
        """
        self.model.eval()
        loss_total = 0.0

        # Reset MONAI metrics
        self.dice_metric.reset()
        self.surface_dist_metric.reset()

        with torch.inference_mode():
            for dict in data_loader:
                # Move data to device
                imgs = dict["image"].to(self.device, non_blocking=True)       # [B, C_img, H, W, D]
                masks = dict["label"].to(self.device, non_blocking=True)     # [B, H, W, D] (integer labels)

                # --- SLIDING WINDOW INFERENCE ---
                # Each batch may contain multiple volumes; we run sliding window per volume.
                # Here, we assume batch size B=1 for 3D volumes or handle one by one:
                # If B>1 with mixed sizes, you might loop over each sample. For simplicity:
                B, C, H, W, D = imgs.shape
                # Create placeholder for aggregated logits
                aggregated_logits = torch.zeros((B, self.num_classes, H, W, D), device=self.device)
                for b in range(B):
                    single_img = imgs[b:b+1]  # [1, C, H, W, D]
                    # Perform sliding window inference on this single volume
                    logits_patch = sliding_window_inference(
                        inputs=single_img,
                        roi_size=self.train_params['shape'],        # e.g. (128,128,64)
                        sw_batch_size=self.train_params.get('sw_batch_size', 1),
                        predictor=lambda x: self.model(x),
                        overlap=self.train_params.get('sw_overlap', 0.25),
                        mode="gaussian"
                    )  # output: [1, num_classes, H, W, D]
                    aggregated_logits[b] = logits_patch

                # Compute loss using aggregated logits
                loss = self.criterion(aggregated_logits, masks)
                loss_total += loss.item()

                # Convert aggregated logits → discrete labels & one-hot encoding
                pred_labels = torch.argmax(aggregated_logits, dim=1, keepdim=True)  # [B, H, W, D]
                pred_onehot = one_hot(pred_labels, num_classes=self.num_classes)  # [B, C, H, W, D]

                # Compute Dice (averaged across classes & batch)
                self.dice_metric(y_pred=pred_onehot, y=masks)

                # Compute Surface Distance (mean symmetric)
                self.surface_dist_metric(y_pred=pred_onehot, y=masks)

        # Aggregate loss over batches
        mean_val_loss = loss_total / len(data_loader)

        # Aggregate and reset MONAI metrics → return scalar floats
        dice_score = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        surf_dist_score = self.surface_dist_metric.aggregate().item()
        self.surface_dist_metric.reset()

        return mean_val_loss, {'dice': dice_score, 'surf_dist': surf_dist_score}

    def save_checkpoint(self, epoch, val_metrics):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model.pth'))
        results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)

        # If current epoch is the best so far, save a copy
        if (self.val_metrics['dice'][-1] >= max(self.val_metrics['dice'])
            and self.val_metrics['surf_dist'][-1] <= min(self.val_metrics['surf_dist'])):
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
            self.best_results = {
                'epoch (0-based)': epoch,
                'train_loss': self.train_losses[-1],
                'val_loss': self.val_losses[-1],
                'val_metrics': val_metrics
            }

        elapsed_time = time() - self.start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        with open(os.path.join(self.output_dir, 'results.txt'), 'w') as f:
            f.write(f'Model size: {self.model_size / 1e6:.2f} M\n')
            f.write(f'Training time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n\n')
            for comment in self.comments:
                f.write(f'{comment}\n')
            f.write(f'\nModel params: {json.dumps(self.model.model_params, indent=4)}\n')
            f.write(f'\nTrain params: {json.dumps(self.train_params, indent=4)}\n')
            f.write(f'\nBest validation results: {json.dumps(self.best_results, indent=4)}\n')
            f.write(f'\n~~~~~~ Test Results ~~~~~~\n')

    def plot_results(self):
        epochs = range(1, len(self.train_losses) + 1)

        # Create a figure with 1 row, 2 columns
        fig, (ax_loss, ax_metric) = plt.subplots(
            1, 2, figsize=(12, 5)
        )  # 1 row, 2 columns layout

        # -----------------------------
        # Left panel: Train & Val Loss
        # -----------------------------
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.plot(
            epochs, self.train_losses,
            label='Train Loss', color='tab:blue'
        )
        ax_loss.plot(
            epochs, self.val_losses,
            label='Val Loss', color='tab:orange'
        )
        ax_loss.legend(loc='upper right')
        ax_loss.set_title('Train & Val Loss')

        # --------------------------------------------
        # Right panel: Val Dice & Val Surface Distance
        # --------------------------------------------
        ax_metric.set_xlabel('Epochs')
        ax_metric.set_ylabel('Val Dice')
        # Plot Val Dice on the left y-axis
        line1, = ax_metric.plot(
            epochs, self.val_metrics['dice'],
            label='Val Dice', color='black'
        )

        # Create a twin y-axis for Surface Distance
        ax_metric2 = ax_metric.twinx()  # share the same x-axis
        ax_metric2.set_ylabel('Val Surface Distance')
        # Plot Val Surface Distance on the right y-axis
        line2, = ax_metric2.plot(
            epochs, self.val_metrics['surf_dist'],
            label='Val Surface Distance', color='tab:green'
        )

        # Combine legends from both axes
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax_metric.legend(lines, labels, loc='upper right')
        ax_metric.set_title('Val Metrics')

        # Adjust layout so subplots don’t overlap
        plt.tight_layout()

        # Save figure to disk
        plt.savefig(os.path.join(self.output_dir, 'losses_and_metrics.png'))
        plt.close(fig)