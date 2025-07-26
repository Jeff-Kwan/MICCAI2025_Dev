from monai.transforms import Transform
from monai.data import MetaTensor
from skimage.morphology import remove_small_objects
from torch import Tensor, tensor

class RemoveSmallObjectsPerClassd(Transform):
    def __init__(self, keys, labels, min_sizes, connectivity=1):
        self.keys = keys
        self.labels = labels
        self.min_sizes = min_sizes
        self.conn = connectivity

    def __call__(self, data):
        for key in self.keys:
            img = data[key].cpu().numpy()
            for lbl, ms in zip(self.labels, self.min_sizes):
                mask = (img == lbl)
                if mask.any():
                    cleaned_mask = remove_small_objects(mask, min_size=ms, connectivity=self.conn)
                    img[mask & (~cleaned_mask)] = 0
        
            if isinstance(data[key], MetaTensor):
                data[key] = MetaTensor(img, affine=data[key].affine, meta=data[key].meta)
            elif isinstance(data[key], Tensor):
                data[key] = tensor(img, dtype=data[key].dtype, device=data[key].device)
            else:
                data[key] = img
        return data