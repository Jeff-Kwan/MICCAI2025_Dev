import matplotlib.pyplot as plt
import torch
from matplotlib.collections import LineCollection
import monai.transforms as mt

# Parameters (similar to your example)
sp_size = (100, 100, 1)  # 3D grid, reduced for speed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create 3D grid
grid = mt.utils.create_grid(spatial_size=sp_size, device=device, backend="torch")

# Setup Rand3DElastic transform
rand_3d_elastic = mt.Rand3DElastic(
    prob=1.0,
    sigma_range=(1.75, 1.75),
    magnitude_range=(12, 12),
    mode="trilinear",
)

# Apply transform to grid
deformed_grid = rand_3d_elastic(grid)

# Choose a central slice (XY plane at center Z)
z_idx = sp_size[2] // 2
xx = deformed_grid[0, :, :, z_idx]
yy = deformed_grid[1, :, :, z_idx]

segs1 = torch.stack((xx, yy), axis=2).cpu().numpy()
segs2 = segs1.transpose(1, 0, 2)

fig, ax = plt.subplots()
ax.add_collection(LineCollection(segs1, color="C0"))
ax.add_collection(LineCollection(segs2, color="C0"))
ax.autoscale()
ax.set_aspect("equal")
plt.title("Rand3DElastic: XY slice (center Z)")
plt.show()
