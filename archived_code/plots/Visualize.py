# viz_seg_figure_demoish.py
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import monai.transforms as mt

# --- paths ---------------------------------------------------
# (keep your structure; change if needed)
IMG_DIR        = "archived_code/plots/images"
GT_DIR         = "archived_code/plots/labels/Validation-Public-Labels"
PRED_DIR_AU5   = "archived_code/plots/labels/AU5_outputs"
PRED_DIR_AU6   = "archived_code/plots/labels/AU6_outputs"
OUT_PATH       = "archived_code/plots/Visualize.pdf"

# --- figure look: Times New Roman + dark background ----------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 20,
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "savefig.facecolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
})

# --- helper: CT windowing (level=40, width=400) ---------------
def window_ct(x, level=40.0, width=400.0):
    lo = level - width / 2.0  # -160
    hi = level + width / 2.0  #  240
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-8)

# --- MONAI pipeline (now loads AU5 + AU6 preds) ---------------
loader = mt.Compose([
    mt.LoadImaged(keys=["pred5", "pred6", "image", "label"], ensure_channel_first=True),
    mt.CropForegroundd(keys=["pred5", "pred6", "image", "label"], source_key="label",
                       margin=(500, 500, 0), allow_smaller=True),
    mt.Orientationd(keys=["pred5", "pred6", "image", "label"], axcodes="RAS", lazy=True),
    mt.Spacingd(keys=["pred5", "pred6", "image", "label"], pixdim=(1.0, 1.0, 1.0),
                mode=['nearest', 'nearest', 'trilinear', 'nearest'], lazy=True),
    mt.CenterSpatialCropd(keys=["pred5", "pred6", "image", "label"], roi_size=(450, 450, -1), lazy=True),
    mt.Rotate90d(keys=["pred5", "pred6", "image", "label"], spatial_axes=(0, 1), k=1, lazy=True),
    mt.EnsureTyped(keys=["pred5", "pred6", "image", "label"],
                   dtype=[np.int16, np.int16, np.float32, np.int16], track_meta=True),
])

# --- cases + metrics (yours) ---------------------------------
nums = ['48', '44', '15', '45', '21']
au5_dice = [0.8026, 0.9552, 0.9287, 0.9307, 0.9606]
au5_nsd  = [0.8326, 0.9891, 0.9855, 0.9677, 0.9956]
au6_dice = [0.7776, 0.8698, 0.9045, 0.9153, 0.9485]
au6_nsd  = [0.8158, 0.9135, 0.9828, 0.9671, 0.9958]

# --- label color map (stable colors across rows) -------------
# Up to ~14 labels; add more colors if your dataset has more IDs.
itksnap_hex = [
    "#00000000",  # 0 = clear/transparent background for plotting
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF",
    "#FFEFD5", "#0000CD", "#CD853F", "#D2B48C", "#66CDAA", "#000080", "#008B8B"
]
itksnap_cmap = ListedColormap(itksnap_hex)

# --- make the panel (4 rows x 4 cols) ------------------------
n_rows, n_cols = len(nums), 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows + 0.5), constrained_layout=False)

# leave generous margins like the demo and reserve space at bottom for column titles
plt.subplots_adjust(left=0.06, right=0.995, top=0.98, bottom=0.08, wspace=0.02, hspace=0.04)

for i, (num, au5_d, au5_n, au6_d, au6_n) in enumerate(zip(nums, au5_dice, au5_nsd, au6_dice, au6_nsd)):
    data = loader({
        "pred5": os.path.join(PRED_DIR_AU5, f"FLARETs_00{num}.nii.gz"),
        "pred6": os.path.join(PRED_DIR_AU6, f"FLARETs_00{num}.nii.gz"),
        "image": os.path.join(IMG_DIR,      f"FLARETs_00{num}.nii.gz"),
        "label": os.path.join(GT_DIR,       f"FLARETs_00{num}.nii.gz"),
    })

    pred5 = data["pred5"].numpy()[0]
    pred6 = data["pred6"].numpy()[0]
    image = data["image"].numpy()[0]
    label = data["label"].numpy()[0]

    # choose slice with the largest number of unique classes (excluding background)
    unique_counts = [len(np.unique(label[:, :, idx][label[:, :, idx] > 0])) for idx in range(label.shape[2])]
    score = [uc**0.5 * np.count_nonzero(label[:, :, idx])**0.25 * (label[:, :, idx] != pred6[:, :, idx]).sum() **0.75
             for idx, uc in enumerate(unique_counts)]
    k = int(np.argmax(score))

    img2d  = window_ct(image[:, :, k])  # windowing to L=40, W=400
    gt2d   = label[:, :, k]
    pr5_2d = pred5[:, :, k]
    pr6_2d = pred6[:, :, k]

    # consistent vmax across GT and both predictions for stable colors
    vmax_all = max(int(gt2d.max()), int(pr5_2d.max()), int(pr6_2d.max()), 1)

    # --- column 1: image -------------------------------------
    ax0 = axes[i, 0]
    ax0.imshow(img2d, cmap="gray", interpolation="nearest")
    ax0.axis("off")

    # --- column 2: ground truth overlay ----------------------
    ax1 = axes[i, 1]
    ax1.imshow(img2d, cmap="gray", interpolation="nearest")
    if gt2d.max() > 0:
        ax1.imshow(np.ma.masked_where(gt2d == 0, gt2d),
                   cmap=itksnap_cmap, alpha=0.85, vmin=1, vmax=vmax_all)
    ax1.axis("off")

    # --- column 3: AU5 prediction overlay + yellow metrics ----
    ax2 = axes[i, 2]
    ax2.imshow(img2d, cmap="gray", interpolation="nearest")
    if pr5_2d.max() > 0:
        ax2.imshow(np.ma.masked_where(pr5_2d == 0, pr5_2d),
                   cmap=itksnap_cmap, alpha=0.85, vmin=1, vmax=vmax_all)
    ax2.axis("off")

    # --- column 4: AU6 prediction overlay --------------------
    ax3 = axes[i, 3]
    ax3.imshow(img2d, cmap="gray", interpolation="nearest")
    if pr6_2d.max() > 0:
        ax3.imshow(np.ma.masked_where(pr6_2d == 0, pr6_2d),
                   cmap=itksnap_cmap, alpha=0.85, vmin=1, vmax=vmax_all)
    ax3.axis("off")

    # row label at top left of the row, spanning ax0 and ax1 if needed
    row_tag = f"Case #FLARETs_{num.zfill(4)} (slice #{k})"
    # Get positions of ax0 and ax1
    pos0 = ax0.get_position()
    pos1 = ax1.get_position()
    # Calculate the span: left of ax0 to right of ax1
    x0 = pos0.x0
    x1 = pos1.x1
    # Place text in figure coordinates, spanning both axes
    fig.text(
        x0, pos0.y1 - 0.01, row_tag,
        ha="left", va="top", fontsize=26, color="white",
        zorder=100, rotation=0,
        bbox=dict(facecolor="black", alpha=0.0, edgecolor="none", pad=0.0),
        wrap=True
    )
    
    # metrics box on AU5 prediction pane, centered horizontally
    ax2.text(0.5, 0.9, f"DSC {au5_d:.4f}, NSD {au5_n:.4f}",
             transform=ax2.transAxes, ha="center", va="center",
             fontsize=26, color="yellow",
             bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", boxstyle="round,pad=0.35"))

    # metrics box on AU6 prediction pane, centered horizontally
    ax3.text(0.5, 0.9, f"DSC {au6_d:.4f}, NSD {au6_n:.4f}",
             transform=ax3.transAxes, ha="center", va="center",
             fontsize=24, color="yellow",
             bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", boxstyle="round,pad=0.35"))

# bottom column titles (centered to columns)
titles = ["Image", "Ground Truth", "Pseudo-Labeler", "Inference Model"]
for col in range(n_cols):
    ax = axes[0, col]
    pos = ax.get_position()
    x = pos.x0 + (pos.x1 - pos.x0) / 2
    fig.text(x, 0.03, titles[col], ha="center", va="center", fontsize=30, color="white")

# save
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved to: {OUT_PATH}")
