from typing import Hashable, Any, Mapping, Sequence, Union
import itertools
import numpy as np
import torch
from skimage.morphology import diameter_closing
from monai.config import KeysCollection
from monai.transforms import MapTransform, KeepLargestConnectedComponent
from skimage.morphology import binary_dilation, ball

class DiameterClosingd(MapTransform):
    """
    Dictionary-based transform applying per-channel diameter closing from scikit-image,
    with optional integer-label support when the input has a single channel.

    Args:
        keys: keys of the corresponding items to be transformed.
        diameter_threshold: int or sequence of ints. If a sequence, its length must equal
            number of channels (for multi-channel) or number of classes (for single-channel label mode).
        connectivity: connectivity parameter passed to `skimage.morphology.diameter_closing`
                      (e.g., in 3D, `connectivity=1` is 6-connectivity, `2` yields 26).
        classes: if None or equal to number of channels, behaves like original per-channel closing.
                 If input has a single channel (C == 1) and `classes` > 1, treats the image as
                 an integer label map with labels 1..classes (0 is background), applies diameter
                 closing to each label's binary mask, and recombines. Overlaps are resolved by
                 giving priority to lower label values.
        allow_missing_keys: if True, missing keys are ignored.
    """

    def __init__(
        self,
        keys: KeysCollection,
        diameter_threshold: Union[int, Sequence[int]],
        connectivity: int = 1,
        classes: Union[None, int] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if classes is not None:
            if not (isinstance(classes, int) and classes >= 1):
                raise ValueError("classes must be a positive integer or None.")
        self.diameter_threshold = diameter_threshold
        self.connectivity = connectivity
        self.classes = classes

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

            # label-mode: single channel + classes > 1
            if n_channels == 1 and self.classes is not None and self.classes > 1:
                label_img = img_np[0]
                # prepare per-class thresholds for labels 1..classes
                if np.isscalar(self.diameter_threshold):
                    thresholds = [self.diameter_threshold] * self.classes
                else:
                    if len(self.diameter_threshold) != self.classes:
                        raise ValueError(
                            f"{self.__class__.__name__}: length of diameter_threshold "
                            f"{len(self.diameter_threshold)} != number of classes {self.classes}"
                        )
                    thresholds = list(self.diameter_threshold)

                # initialize output label image (background=0)
                out_label = np.zeros_like(label_img)

                for label in range(1, self.classes + 1):
                    thr = thresholds[label - 1]
                    bin_mask = label_img == label
                    if not np.any(bin_mask):
                        continue  # nothing to do for this label
                    # diameter_closing expects an image; feed binary mask (as uint8 or bool)
                    closed = diameter_closing(
                        bin_mask.astype(np.uint8),
                        diameter_threshold=thr,
                        connectivity=self.connectivity,
                    )
                    closed_mask = closed.astype(bool)
                    # set label only where output is still background to enforce priority
                    to_set = closed_mask & (out_label == 0)
                    out_label[to_set] = label

                out = np.expand_dims(out_label, axis=0)

            else:
                # standard per-channel mode (including single-channel intensity or classes==n_channels)
                if self.classes is not None and self.classes != n_channels:
                    raise ValueError(
                        f"{self.__class__.__name__}: classes ({self.classes}) must be equal to number of channels ({n_channels}) "
                        f"when not in single-channel label mode."
                    )
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

                out_channels = []
                for c, thr in enumerate(thresholds):
                    ch = img_np[c]
                    out_ch = diameter_closing(
                        ch, diameter_threshold=thr, connectivity=self.connectivity
                    )
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


class DilatedForegroundd(MapTransform):
    def __init__(
        self,
        keys,
        dilation: int = 8,
        connectivity: int | None = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.dilation = dilation
        self.connectivity = connectivity
        self._selem = ball(dilation)
        self.keep_largest = KeepLargestConnectedComponent(connectivity=connectivity)

    def _bounding_box(self, mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        return tuple(slice(int(mn), int(mx) + 1) for mn, mx in zip(mins, maxs))

    @staticmethod
    def _zero_outside_expanded(arr: np.ndarray, expanded_slices: tuple):
        """
        Zero out all voxels outside expanded_slices in-place, without constructing a full mask.
        """
        ranges = []
        for sl in expanded_slices:
            ranges.append([
                ('before', slice(None, sl.start)),
                ('inside', slice(sl.start, sl.stop)),
                ('after', slice(sl.stop, None)),
            ])
        for combo in itertools.product(*ranges):
            if all(part[0] == 'inside' for part in combo):
                continue
            sel = tuple(part[1] for part in combo)
            arr[sel] = 0

    def __call__(self, data: Mapping[Hashable, Any]) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            arr = d[key]
            is_torch = isinstance(arr, torch.Tensor)

            if is_torch:
                device = arr.device
                arr_np = arr.cpu().numpy()
            else:
                arr_np = arr  # assume numpy array

            # get bounding box of positive foreground (arr_np > 0)
            bbox = self._bounding_box(arr_np > 0)
            if bbox is None:
                continue  # nothing to keep

            # expand bbox by dilation, clipped to volume
            expanded_slices = []
            for dim, sl in enumerate(bbox):
                start = max(0, sl.start - self.dilation)
                stop = min(arr_np.shape[dim], sl.stop + self.dilation)
                expanded_slices.append(slice(start, stop))
            expanded_slices = tuple(expanded_slices)

            # get subvolume in expanded region
            sub_vol = arr_np[expanded_slices]

            # compute foreground in subvolume, dilate, and keep largest component
            sub_fg = sub_vol > 0
            dilated = binary_dilation(sub_fg, footprint=self._selem)
            largest = self.keep_largest(dilated)

            # zero out everything in the subvolume except the kept component (in-place)
            sub_vol[~largest] = 0

            # zero everything outside expanded region (in-place)
            self._zero_outside_expanded(arr_np, expanded_slices)

            if is_torch:
                # push back into original tensor in-place when possible
                updated = torch.from_numpy(arr_np)
                if updated.device != device or updated.dtype != arr.dtype:
                    updated = updated.to(device=device, dtype=arr.dtype)
                try:
                    arr.copy_(updated)  # in-place update original tensor
                    d[key] = arr
                except RuntimeError:
                    # fallback if in-place fails
                    d[key] = updated
            else:
                d[key] = arr_np
        return d