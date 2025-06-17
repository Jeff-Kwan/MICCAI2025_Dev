import os
import json
import time
import torch
from torch.nn.functional import interpolate
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm
import matplotlib.pyplot as plt
import monai.metrics as mm
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference

class VAETrainer:
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

        # VAE params
        beta = train_params.get('beta', 1.0)
        self.beta = torch.linspace(beta[0], beta[1], beta[2], device=self.device)
        if beta[2] < train_params['epochs'][0]:
            self.beta = torch.cat([self.beta, self.beta[-1].repeat(train_params['epochs'][0] - beta[2])])

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
        self.dice_metric = mm.DiceMetric(include_background=False)
        if self.local_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            self.vae_losses = []
            self.model_losses = []
            self.val_losses = []
            self.val_metrics = {'dice': []}
            self.best_results = {}
            self.model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.start_time = None

    def kl_div_normal(self, mu, logvar):
        # Sum over latent dim - channels (1); mean over batch and img dimensions
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def kl_gaussian(self, mu, logvar, mu_target, logvar_target):
        # mu, logvar: [batch, latent_dim]
        # mu_target, logvar_target: [batch, latent_dim] or [latent_dim]
        var = logvar.exp()
        var_target = logvar_target.exp()
        kl = 0.5 * (logvar_target - logvar + (var + (mu - mu_target).pow(2)) / var_target - 1)
        # Sum over latent dimension, mean over batch
        return kl.sum(dim=1).mean()

    def train(self, train_loader, val_loader=None):
        if self.local_rank == 0:
            self.start_time = time.time()

        epochs = self.train_params['epochs']
        agg_steps = self.train_params['aggregation']

        for epoch in range(epochs):
            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            self.model.train()
            running_vae_loss = 0.0
            running_model_loss = 0.0
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
                    pred, mu_hat, log_var_hat, prior_pred, mu, log_var = self.model(imgs, masks)
                    # Loss:
                        # 1. Reconstruction loss (pred vs masks)
                        # 2. Reconstruction loss (prior_pred vs masks)
                        # 3. KL divergence between mu, log_var and prior
                        # 4. KL between mu_hat, log_var_hat and mu, log_var
                    label = interpolate(masks.float(), scale_factor=0.5, mode='nearest').long()
                    vae_recon_loss = self.criterion(prior_pred, label)
                    model_recon_loss = self.criterion(pred, label)
                    loss = model_recon_loss + vae_recon_loss +\
                            self.beta[epoch] * self.kl_div_normal(mu, log_var) +\
                            self.kl_gaussian(mu_hat, log_var_hat, mu.detach(), log_var.detach())

                loss.backward()
                running_loss += loss.item()
                running_vae_loss += vae_recon_loss.item()
                running_model_loss += model_recon_loss.item()


                if ((i + 1) % agg_steps == 0) or (i + 1 == len(train_loader)):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.local_rank == 0:
                    loop.set_postfix({'Norm': grad_norm.item(), 'Loss': loss.item()})

            self.scheduler.step()

            val_loss, metrics = self.evaluate(val_loader)
            if self.world_size > 1:
                torch.cuda.synchronize(self.device)
                dist.barrier()
            if self.local_rank == 0 and val_loader is not None:
                self.vae_losses.append(running_vae_loss / len(train_loader))
                self.model_losses.append(running_model_loss / len(train_loader))
                self.val_losses.append(val_loss)
                self.val_metrics['dice'].append(metrics['dice'])
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"VAE Loss: {running_vae_loss / len(train_loader):.5f} | "
                      f"Model Loss: {running_model_loss / len(train_loader):.5f} | "
                      f"Val Loss: {val_loss:.5f} | "
                      f"Val Dice: {metrics['dice']:.5f}")
                self.plot_results()
                self.save_checkpoint(epoch, metrics)


    def train_prior(self, train_loader, epochs, agg_steps=1):
        if self.local_rank == 0:
            self.start_time = time.time()

        for epoch in range(epochs):
            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            self.model.train()
            running_vae_loss = 0.0
            grad_norm = torch.tensor(0.0, device=self.device)

            loop = tqdm.tqdm(train_loader,
                             desc=f"[Rank {self.local_rank}] Epoch {epoch+1}/{epochs}",
                             disable=(self.local_rank!=0))
            self.optimizer.zero_grad()

            for i, batch in enumerate(loop):
                masks = batch['label'].to(self.device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=self.precision):
                    prior_pred, mu, log_var = self.model.module.vae_prior(masks)
                    label = interpolate(masks.float(), scale_factor=0.5, mode='nearest').long()
                    loss = self.criterion(prior_pred, label) +\
                            self.beta[epoch] * self.kl_div_normal(mu, log_var)

                loss.backward()
                running_vae_loss += loss.item()

                if ((i + 1) % agg_steps == 0) or (i + 1 == len(train_loader)):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.local_rank == 0:
                    loop.set_postfix({'Norm': grad_norm.item(), 'Loss': loss.item()})

            self.scheduler.step()

            if self.world_size > 1:
                torch.cuda.synchronize(self.device)
                dist.barrier()
            if self.local_rank == 0:
                self.vae_losses.append(running_vae_loss / len(train_loader))
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"VAE Loss: {running_vae_loss / len(train_loader):.5f} | ")
                self.plot_vae_results()
                torch.save(self.model.module.vae_prior.state_dict(),
                            os.path.join(self.output_dir, 'vae_prior.pth'))


    def train_posterior(self, train_loader, val_loader, epochs, agg_steps=1):
        if self.local_rank == 0:
            self.start_time = time.time()

        for epoch in range(epochs):
            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            # Freeze VAE prior
            for param in self.model.module.vae_prior.parameters():
                param.requires_grad = False
            self.model.train()

            running_model_loss = 0.0
            grad_norm = torch.tensor(0.0, device=self.device)

            loop = tqdm.tqdm(train_loader,
                             desc=f"[Rank {self.local_rank}] Epoch {epoch+1}/{epochs}",
                             disable=(self.local_rank!=0))
            self.optimizer.zero_grad()

            for i, batch in enumerate(loop):
                imgs = batch['image'].to(self.device, non_blocking=True)
                masks = batch['label'].to(self.device, non_blocking=True)

                with torch.autocast(device_type='cuda', dtype=self.precision):
                    with torch.no_grad():
                        mu, log_var = self.model.module.vae_prior.encode(masks)
                        prior_z = self.model.module.vae_prior.reparameterize(mu, log_var)
                        _, latent_priors = self.model.module.vae_prior.decode(prior_z)
                
                    mu_hat, log_var_hat, skips = self.model.module.img_encode(imgs)
                    skips = [s.detach().clone().requires_grad_() for s in skips]
                    latent_priors = [lp.clone().requires_grad_() for lp in latent_priors]
                    pred = self.model.module.decode(prior_z.clone().requires_grad_(), skips, latent_priors)

                    label = interpolate(masks.float(), scale_factor=0.5, mode='nearest').long()
                    loss = self.criterion(pred, label) +\
                        self.kl_gaussian(mu_hat, log_var_hat, mu, log_var)

                loss.backward()
                running_model_loss += loss.item()


                if ((i + 1) % agg_steps == 0) or (i + 1 == len(train_loader)):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.local_rank == 0:
                    loop.set_postfix({'Norm': grad_norm.item(), 'Loss': loss.item()})

            self.scheduler.step()

            val_loss, metrics = self.evaluate(val_loader)
            if self.world_size > 1:
                torch.cuda.synchronize(self.device)
                dist.barrier()
            if self.local_rank == 0 and val_loader is not None:
                self.model_losses.append(running_model_loss / len(train_loader))
                self.val_losses.append(val_loss)
                self.val_metrics['dice'].append(metrics['dice'])
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Model Loss: {running_model_loss / len(train_loader):.5f} | "
                      f"Val Loss: {val_loss:.5f} | "
                      f"Val Dice: {metrics['dice']:.5f}")
                self.plot_posterior_results()
                self.save_checkpoint(epoch, metrics)

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
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
                
                torch.cuda.empty_cache()    # clear cache after inference

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

        # Aggregate loss and sample count across all ranks
        if self.world_size > 1:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)

        total_loss = loss_sum.item() / max(sample_count.item(), 1)
        total_dice = float(self.dice_metric.aggregate())
        return total_loss, {'dice': total_dice}

    def save_checkpoint(self, epoch: int, val_metrics: dict):
        # Save last
        state_dict = (self.model.module.state_dict()
                if isinstance(self.model, DDP) else self.model.state_dict())
        torch.save(state_dict, os.path.join(self.output_dir, 'model.pth'))
        history = {
            'vae_losses': self.vae_losses,
            'model_losses': self.model_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(history, f, indent=4)

        # Save best
        if self.val_metrics['dice'][-1] == max(self.val_metrics['dice']):
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))
            self.best_results = {
                'epoch': epoch,
                'vae_loss': self.vae_losses[-1],
                'model_loss': self.model_losses[-1],
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
            f.write(f"Epoch {epoch+1} results:\n")
            f.write(f"VAE Loss: {self.vae_losses[-1]:.5f}; Model Loss: {self.model_losses[-1]:.5f}; Val Loss: {self.val_losses[-1]:.5f}; Val Dice: {self.val_metrics['dice'][-1]:.5f}\n\n")
            for c in self.comments:
                f.write(c + "\n")
            f.write(f"\nModel params: {json.dumps(self.model.module.model_params, indent=4)}\n")
            f.write(f"\nTrain params: {json.dumps(self.train_params, indent=4)}\n")
            f.write(f"\nBest results: {json.dumps(self.best_results, indent=4)}\n")

    def plot_results(self):
        epochs = range(1, len(self.val_losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss curve
        ax1.plot(epochs, self.vae_losses, label='VAE', color='black')
        ax1.plot(epochs, self.model_losses, label='Model', color='blue')
        ax1.plot(epochs, self.val_losses, label='Val', color='orange')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.legend(); ax1.set_title('Loss')

        # Dice curve
        ax2.plot(epochs, self.val_metrics['dice'], label='Val Dice', color='orange')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice')
        ax2.legend(); ax2.set_title('Validation Dice')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close(fig)

    def plot_vae_results(self):
        epochs = range(1, len(self.vae_losses) + 1)
        plt.figure()
        plt.plot(epochs, self.vae_losses, label='VAE Loss', color='black')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('VAE Training Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'vae_training_loss.png'))
        plt.close()


    def plot_posterior_results(self):
        epochs = range(1, len(self.val_losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss curve
        ax1.plot(epochs, self.model_losses, label='Model', color='blue')
        ax1.plot(epochs, self.val_losses, label='Val', color='orange')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.legend(); ax1.set_title('Loss')

        # Dice curve
        ax2.plot(epochs, self.val_metrics['dice'], label='Val Dice', color='orange')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice')
        ax2.legend(); ax2.set_title('Validation Dice')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_posterior_loss.png'))
        plt.close(fig)