import monai.transforms as mt
import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# Load your two label volumes
loader = mt.Compose([
    mt.LoadImaged(keys=["pred", "image", "label"], ensure_channel_first=True),
    mt.CropForegroundd(keys=["pred", "image", "label"], source_key="label", margin=(500, 500, 0), allow_smaller=True),
    mt.Orientationd(keys=["pred", "image", "label"], axcodes="RAS", lazy=True),
    mt.Spacingd(keys=["pred", "image", "label"], pixdim=(1.0, 1.0, 1.0), mode=['nearest', 'trilinear', 'nearest'], lazy=True),
    mt.Rotate90d(keys=["pred", "image", "label"], spatial_axes=(0, 1), k=1, lazy=True),
    mt.EnsureTyped(keys=["pred", "image", "label"], dtype=[np.int8, np.float32, np.int8], track_meta=True),
])
nums = ['48', '07', '22', '15', '21']
dice = [0.7566, 0.8689, 0.9007, 0.9137, 0.9465]
nsd = [0.7057, 0.8742, 0.9060, 0.8438, 0.9732]
fig, axes = plt.subplots(len(nums), 3, figsize=(15, 5*len(nums)))

col_titles = ["Image", "Ground Truth", "Prediction"]
for ax, title in zip(axes[0], col_titles):
    ax.set_title(f"{title}", fontsize=26, pad=12)

# Leave room on the left
plt.subplots_adjust(left=0.1, right=1.0, top=1.0, bottom=0.0)

for i, num in enumerate(nums):
    data = loader({
        "pred": rf"archived_code\plots\labels\au5_post_output\FLARETs_00{num}.nii.gz",
        "image": rf"archived_code\plots\images\FLARETs_00{num}.nii.gz",
        "label": rf"archived_code\plots\labels\Validation-Public-Labels\FLARETs_00{num}.nii.gz"
    })

    pred  = data["pred"].numpy()[0]
    image = data["image"].numpy()[0]
    label = data["label"].numpy()[0]

    # Choose slice with the most going on
    eventful = np.sum(label > 0, axis=(0, 1)) #* np.sum(label != pred, axis=(0, 1))
    ind = np.argmax(eventful)
    pred  = pred[:, :, ind]
    image = image[:, :, ind]
    label = label[:, :, ind]

    # Normalize image for better visualization
    image = (image - image.min()) / (image.max() - image.min())

    # Plot panels
    axes[i, 0].imshow(image, cmap="gray")
    axes[i, 1].imshow(image, cmap="gray")
    axes[i, 1].imshow(
        np.ma.masked_where(label == 0, label),
        alpha=0.8, cmap="jet", vmin=1, vmax=label.max()
    )
    axes[i, 2].imshow(image, cmap="gray")
    axes[i, 2].imshow(
        np.ma.masked_where(pred == 0, pred),
        alpha=0.8, cmap="jet", vmin=1, vmax=max(label.max(), pred.max())
    )

    # Row heading on the left
    row_label = f"Case #{num}"
    axes[i, 0].text(
        -0.08, 0.5, row_label,
        transform=axes[i, 0].transAxes,
        fontsize=24, va='center', ha='left', rotation='vertical'
    )
    results = f"DSC {dice[i]}\nNSD {nsd[i]}"
    axes[i, 2].text(
        0.8, 0.9, results,
        transform=axes[i, 2].transAxes,
        fontsize=20, va='center', ha='center', color='yellow'
    )

    # Turn axes off
    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()
plt.savefig("archived_code/plots/Visualize.pdf", format='pdf')