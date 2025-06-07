import torch
import tqdm
import os
import json
import matplotlib.pyplot as plt
from time import time

import monai.metrics as mm
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
            'surface_dice': [],  # now tracking normalized surface dice
        }
        self.best_results = {}

        self.precision = torch.bfloat16 if train_params["autocast"] else torch.float32

        os.makedirs(output_dir, exist_ok=True)
        self.model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Number of segmentation classes (including background)
        self.num_classes = self.train_params['num_classes']

        # MONAI Dice metric: average over batch and classes, ignore background (channel=0)
        self.dice_metric = mm.DiceMetric(include_background=False)

        # MONAI Surface Dice metric: normalized surface dice (NSD)
        # you can override thresholds via train_params['surface_dice_class_thresholds']
        thresholds = self.train_params.get(
            'surface_dice_class_thresholds',
            [1.0] * self.num_classes
        )
        self.surface_dice_metric = mm.SurfaceDiceMetric(
            class_thresholds=thresholds,
            include_background=False,
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

            for i, batch in enumerate(p_bar):
                imgs = batch["image"].to(self.device, non_blocking=True)
                masks = batch["label"].to(self.device, non_blocking=True)

                with torch.autocast('cuda', dtype=self.precision):
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, masks)
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
            self.val_metrics['surface_dice'].append(metrics['surface_dice'])

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {self.train_losses[-1]:.5f} | "
                f"Val Loss: {self.val_losses[-1]:.5f} | "
                f"Val Dice: {metrics['dice']:.5f} | "
                f"Val Surface Dice: {metrics['surface_dice']:.5f}"
            )

            # Save results, checkpoint, etc.
            self.plot_results()
            self.save_checkpoint(epoch, metrics)

    def evaluate(self, data_loader):
        """
        Runs one full pass over data_loader, computes the mean validation loss,
        and then uses MONAI's DiceMetric and SurfaceDiceMetric to compute
        multi-class Dice and normalized surface dice (NSD) scores over all batches.
        """
        self.model.eval()
        loss_total = 0.0

        # Reset MONAI metrics
        self.dice_metric.reset()
        self.surface_dice_metric.reset()

        with torch.inference_mode():
            p_bar = tqdm.tqdm(data_loader, desc='Validation')
            for batch in p_bar:
                imgs = batch["image"].to(self.device, non_blocking=True)      # [B, C_img, H, W, D]
                masks = batch["label"].to(self.device, non_blocking=True)    # [B, H, W, D] (integer labels)

                with torch.autocast('cuda', dtype=self.precision):
                    # sliding window inference per volume
                    B, C, H, W, D = imgs.shape
                    aggregated_logits = torch.zeros((B, self.num_classes, H, W, D), device=self.device)
                    for b in range(B):
                        single = imgs[b:b+1]
                        logits_patch = sliding_window_inference(
                            inputs=single,
                            roi_size=self.train_params['shape'],
                            sw_batch_size=self.train_params.get('sw_batch_size', 1),
                            predictor=lambda x: self.model(x),
                            overlap=self.train_params.get('sw_overlap', 0.25),
                            mode="gaussian"
                        )
                        aggregated_logits[b] = logits_patch

                    loss = self.criterion(aggregated_logits, masks)
                loss_total += loss.item()

                # discrete labels & one-hot for metrics
                pred_labels = torch.argmax(aggregated_logits, dim=1, keepdim=True)
                pred_onehot = one_hot(pred_labels, num_classes=self.num_classes)
                masks_onehot = one_hot(masks, num_classes=self.num_classes)

                # update metrics
                self.dice_metric(y_pred=pred_onehot, y=masks_onehot)
                self.surface_dice_metric(y_pred=pred_onehot, y=masks_onehot)

        mean_val_loss = loss_total / len(data_loader)

        # aggregate & reset
        dice_score = self.dice_metric.aggregate().item()
        surface_score = self.surface_dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.surface_dice_metric.reset()

        return mean_val_loss, {'dice': dice_score, 'surface_dice': surface_score}

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
        if self.val_metrics['dice'][-1] >= max(self.val_metrics['dice']):
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
            self.best_results = {
                'epoch (0-based)': epoch,
                'train_loss': self.train_losses[-1],
                'val_loss': self.val_losses[-1],
                'val_metrics': val_metrics
            }

        elapsed = time() - self.start_time
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        with open(os.path.join(self.output_dir, 'results.txt'), 'w') as f:
            f.write(f'Model size: {self.model_size / 1e6:.2f} M\n')
            f.write(f'Training time: {int(hrs):02}:{int(mins):02}:{int(secs):02}\n\n')
            for comment in self.comments:
                f.write(f'{comment}\n')
            f.write(f'\nModel params: {json.dumps(self.model.model_params, indent=4)}\n')
            f.write(f'\nTrain params: {json.dumps(self.train_params, indent=4)}\n')
            f.write(f'\nBest validation results: {json.dumps(self.best_results, indent=4)}\n')
            f.write(f'\n~~~~~~ Test Results ~~~~~~\n')

    def plot_results(self):
        epochs = range(1, len(self.train_losses) + 1)

        fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(12, 5))

        # Train & Val Loss
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.plot(epochs, self.train_losses,    label='Train Loss', color='tab:blue')
        ax_loss.plot(epochs, self.val_losses,      label='Val Loss',   color='tab:orange')
        ax_loss.legend(loc='upper right')
        ax_loss.set_title('Train & Val Loss')

        # Val Dice & Surface Dice
        ax_metric.set_xlabel('Epochs')
        ax_metric.set_ylabel('Metric Value')
        line1, = ax_metric.plot(epochs, self.val_metrics['dice'],
                                label='Val Dice',         color='black')
        line2, = ax_metric.plot(epochs, self.val_metrics['surface_dice'],
                                label='Val Surface Dice', color='tab:green')
        ax_metric.legend([line1, line2], [line1.get_label(), line2.get_label()],
                         loc='upper right')
        ax_metric.set_title('Val Metrics')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'losses_and_metrics.png'))
        plt.close(fig)
