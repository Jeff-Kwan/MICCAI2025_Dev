import monai.transforms as mt
import napari

# Load your two label volumes
loader = mt.Compose([
    mt.LoadImaged(keys=["vol1", "vol2"], ensure_channel_first=True),
    # mt.CropForegroundd(keys=["vol1", "vol2"], source_key="vol2"),
    # mt.KeepLargestConnectedComponentd(keys=["vol1"]),
    mt.Orientationd(keys=["vol1", "vol2"], axcodes="RAS"),
    mt.Spacingd(keys=["vol1", "vol2"], pixdim=(1.5, 1.5, 1.5), mode=['nearest', 'nearest']),
])
num = '42'
data = loader({
    "vol1": rf"archived_code\plots\labels\attnunet3_output\FLARETs_00{num}.nii.gz",
    "vol2": rf"archived_code\plots\labels\Validation-Public-Labels\FLARETs_00{num}.nii.gz"
})
vol1 = data["vol1"].numpy().astype(int)
vol2 = data["vol2"].numpy().astype(int)
diff = (vol1 != vol2).astype(int)

viewer = napari.Viewer(ndisplay=3)

# Prediction
# viewer.add_labels(
#     vol1,
#     name="Prediction",
#     opacity=0.2,
# )

# Label
viewer.add_labels(
    vol2,
    name="Ground Truth",
    opacity=0.2,
)

# Difference mask, binary
viewer.add_labels(
    diff,
    name="Differences",
    opacity=0.8,
)

napari.run()
