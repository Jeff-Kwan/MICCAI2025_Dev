import monai.transforms as mt
import napari
import torch

# Load your two label volumes
loader = mt.Compose([
    mt.LoadImaged(keys=["vol1", "vol2"], ensure_channel_first=True),
    mt.CropForegroundd(keys=["vol1", "vol2"], source_key="vol2"),
    mt.KeepLargestConnectedComponentd(keys=["vol1"]),
    mt.Orientationd(keys=["vol1", "vol2"], axcodes="RAS"),
    mt.Spacingd(keys=["vol1", "vol2"], pixdim=(1.5, 1.5, 1.5), mode=['nearest', 'nearest']),
])
data = loader({
    "vol1": r"archived_code\plots\labels\val_outputs\FLARETs_0031.nii.gz",
    "vol2": r"archived_code\plots\labels\Validation-Public-Labels\FLARETs_0031.nii.gz"
})
vol1 = data["vol1"].numpy().astype(int)
vol2 = data["vol2"].numpy().astype(int)
diff = (vol1 != vol2).astype(int)

viewer = napari.Viewer(ndisplay=3)

# Prediction
# viewer.add_labels(
#     vol1,
#     name="Seg A",
#     opacity=0.4,
# )

# Label
viewer.add_labels(
    vol2,
    name="Seg B",
    opacity=0.8,
)

# Difference mask, binary (0=transparent, 1=magenta)
viewer.add_labels(
    diff,
    name="Differences",
    opacity=0.4,
)

napari.run()
