import torch
from monai.losses import DiceFocalLoss
from VAEDual import VAEPosterior


def find_missing_gradients(model):
    """
    Returns a list of parameter names that either have no grad
    or whose grad is entirely zero.
    """
    missing = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(f"{name} (no grad)")
        else:
            # Check for all-zero gradient tensor
            if torch.all(param.grad == 0):
                missing.append(f"{name} (zero grad)")
    return missing


def training_step(model, imgs, masks, optimizer, criterion, kl_div_normal, mse):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    pred, mu_hat, log_var_hat, prior_pred, mu, log_var = model(imgs, masks)

    # Composite loss
    vae_recon_loss   = criterion(prior_pred, masks)
    model_recon_loss = criterion(pred,       masks)
    loss = (
        model_recon_loss
        + vae_recon_loss
        + kl_div_normal(mu, log_var)
        + mse(mu_hat, mu.detach().clone().requires_grad_())
        + mse(log_var_hat, log_var.detach().clone().requires_grad_())
    )

    # Backward
    loss.backward()

    # Report missing grads
    missing = find_missing_gradients(model)
    if missing:
        print("⚠️ Parameters with missing or zero gradients:")
        for entry in missing:
            print("   -", entry)
    else:
        print("✅ All parameters received non-zero gradients.")

    optimizer.step()

def kl_div_normal(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

criterion = DiceFocalLoss(
    include_background=True, 
    to_onehot_y=True, 
    softmax=True)

mse = torch.nn.MSELoss()

# Example usage
B, C, H, W, D = 8, 14, 48, 48, 48
x = torch.randn(B, 1, H, W, D).cuda()
y = torch.randint(0, C, (B, 1, H, W, D)).cuda()

model = VAEPosterior({
        "out_channels": 14,
        "channels":     [32, 64, 128, 256],
        "convs":        [32, 48, 64, 32],
        "layers":       [1, 1, 1, 1],
        "dropout":      0.1,
        "stochastic_depth": 0.1
    }).cuda().train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Run a single training step
training_step(model, x, y, optimizer, criterion, kl_div_normal, mse)
