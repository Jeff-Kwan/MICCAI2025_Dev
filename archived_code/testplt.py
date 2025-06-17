import matplotlib.pyplot as plt

def plot_results(vae_losses, model_losses, val_losses, val_metrics, output_dir):
    epochs = range(1, len(val_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(epochs, vae_losses, label='VAE', color='black')
    ax1.plot(epochs, model_losses, label='Model', color='blue')
    ax1.plot(epochs, val_losses, label='Val', color='orange')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.set_title('Loss')

    # Dice curve
    ax2.plot(epochs, val_metrics['dice'], label='Val Dice', color='orange')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice')
    ax2.legend(); ax2.set_title('Validation Dice')

    plt.tight_layout()
    plt.show()


val_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
vae_losses = [0.6, 0.5, 0.4, 0.3]
model_losses = [0.55, 0.45, 0.35, 0.25, 0.15]

val_metrics = {
    'dice': [0.6, 0.65, 0.7, 0.75, 0.8],
}
output_dir = 'output'
plot_results(vae_losses, model_losses, val_losses, val_metrics, output_dir)