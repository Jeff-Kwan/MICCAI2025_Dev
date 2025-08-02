from typing import Sequence, Union, Hashable, Mapping, Any
import numpy as np
import torch
from skimage.morphology import diameter_closing
from monai.transforms import MapTransform, KeepLargestConnectedComponent
from monai.config import KeysCollection


class DiameterClosingd(MapTransform):
    """
    Dictionary-based transform applying per-channel diameter closing from scikit-image.

    Args:
        keys: keys of the corresponding items to be transformed.
        diameter_threshold: int or sequence of ints. If a sequence, its length must equal number of channels.
        connectivity: connectivity parameter passed to `skimage.morphology.diameter_closing`
                      (e.g., in 3D, `connectivity=1` is 6-connectivity, `2` yields 26).
        allow_missing_keys: if True, missing keys are ignored.
    """

    def __init__(
        self,
        keys: KeysCollection,
        diameter_threshold: Union[int, Sequence[int]],
        connectivity: int = 1,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.diameter_threshold = diameter_threshold
        self.connectivity = connectivity

    def __call__(self, data: Mapping[Hashable, Any]) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            is_tensor = isinstance(img, torch.Tensor)
            # convert to numpy array for skimage
            if is_tensor:
                orig_device = img.device
                orig_dtype = img.dtype
                img_np = img.detach().cpu().numpy()
            else:
                orig_dtype = getattr(img, "dtype", None)
                img_np = np.asarray(img)

            if img_np.ndim < 1:
                raise ValueError(
                    f"{self.__class__.__name__}: input for key '{key}' must have a channel dimension."
                )

            n_channels = img_np.shape[0]
            # prepare per-channel thresholds
            if np.isscalar(self.diameter_threshold):
                thresholds = [self.diameter_threshold] * n_channels
            else:
                if len(self.diameter_threshold) != n_channels:
                    raise ValueError(
                        f"{self.__class__.__name__}: length of diameter_threshold "
                        f"{len(self.diameter_threshold)} != number of channels {n_channels}"
                    )
                thresholds = list(self.diameter_threshold)

            # apply diameter closing per channel
            out_channels = []
            for c, thr in enumerate(thresholds):
                ch = img_np[c]
                out_ch = diameter_closing(ch, diameter_threshold=thr, connectivity=self.connectivity)
                out_channels.append(out_ch)
            out = np.stack(out_channels, axis=0)

            # cast back to original type / tensor
            if is_tensor:
                out_tensor = torch.as_tensor(out, device=orig_device)
                if out_tensor.dtype != orig_dtype:
                    out_tensor = out_tensor.to(orig_dtype)
                d[key] = out_tensor
            else:
                if orig_dtype is not None:
                    try:
                        out = out.astype(orig_dtype, copy=False)
                    except Exception:
                        pass
                d[key] = out
        return d


class DiameterForegroundd(MapTransform):
    '''Expects integer labels'''
    def __init__(
        self,
        keys: KeysCollection,
        diameter_threshold: Union[int, Sequence[int]],
        connectivity: int = 1,
        allow_missing_keys: bool = False):
        super().__init__()

        self.diameter_threshold = diameter_threshold
        self.connectivity = connectivity
        self.diameter_closing = DiameterClosingd(
            keys=keys,
            diameter_threshold=diameter_threshold,
            connectivity=connectivity,
            allow_missing_keys=allow_missing_keys)
        self.keeplargest = KeepLargestConnectedComponent(connectivity=3)
        
    def __call__(self, data: Mapping[Hashable, Any]) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            fg = d[key] > 0
            fg = self.diameter_closing(fg)
            fg = self.keeplargest(fg).bool()
            d[key][~fg] = 0
        return d
