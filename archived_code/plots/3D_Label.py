import monai.transforms as mt
import napari
import numpy as np

# Load your two label volumes
loader = mt.Compose([
    mt.LoadImaged(keys=["vol1", "vol2"], ensure_channel_first=True),
    # mt.CropForegroundd(keys=["vol1", "vol2"], source_key="vol2"),
    # mt.KeepLargestConnectedComponentd(keys=["vol1"]),
    mt.Orientationd(keys=["vol1", "vol2"], axcodes="RAS"),
    mt.Spacingd(keys=["vol1", "vol2"], pixdim=(1.5, 1.5, 1.5), mode=['nearest', 'nearest']),
])
clean_small = True
organ = np.array([6])
num = '39'
data = loader({
    "vol1": rf"archived_code\plots\labels\attnunet3_output\FLARETs_00{num}.nii.gz",
    "vol2": rf"archived_code\plots\labels\Validation-Public-Labels\FLARETs_00{num}.nii.gz"
})
vol1 = data["vol1"].numpy().astype(int)
vol2 = data["vol2"].numpy().astype(int)

fp = ((vol1 > vol2) & (vol2 == 0)).astype(int)
fn = ((vol1 < vol2) & (vol1 == 0)).astype(int)
wrong_class = ((vol1 != vol2) & (vol1 > 0) & (vol2 > 0)).astype(int)

if organ is not None:
    vol1 = vol1 * np.isin(vol1, organ)
    vol2 = vol2 * np.isin(vol2, organ)
    fp = fp * (vol1 > 0)
    fn = fn * (vol2 > 0)

if clean_small:
    remove_small = mt.RemoveSmallObjects(min_size=20)
    fp = remove_small(fp)
    fn = remove_small(fn)
    wrong_class = remove_small(wrong_class)


viewer = napari.Viewer(ndisplay=3)

# Prediction
viewer.add_labels(
    vol1,
    name="Prediction",
    opacity=0.4,
    blending='translucent_no_depth',
)

# Label
viewer.add_labels(
    vol2,
    name="Ground Truth",
    opacity=0.4,
    blending='translucent_no_depth',
)

# False Positive mask, binary
viewer.add_labels(
    fp,
    name="False Positives",
    opacity=0.8,
    blending='opaque'
)
viewer.add_labels(
    fn,
    name="False Negatives",
    opacity=0.8,
    blending='opaque'
)
viewer.add_labels(
    wrong_class,
    name="Wrong Class",
    opacity=0.8,
    blending='opaque'
)


napari.run()
