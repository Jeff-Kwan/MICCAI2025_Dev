from typing import Hashable, Any, Mapping
from skimage.morphology import binary_closing
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform, KeepLargestConnectedComponent

class BinaryClosingd(MapTransform):
    """
    Expects single channel integer labels, numpy array format.
    """

    def __init__(
        self,
        keys: KeysCollection,
        width: int = 4,
        classes: int = 14,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.width = width
        self.classes = classes
        self.footprint = np.ones((width, width, width), dtype=bool)  # Cubic footprint

    def __call__(self, data: Mapping[Hashable, Any]) -> dict:
        for key in self.key_iterator(data):
            for class_index in range(1, self.classes):
                fg = (data[key] == class_index).squeeze(0)
                fg = binary_closing(fg, footprint=self.footprint)
                fg = np.expand_dims(fg, axis=0)
                data[key][(fg & (data[key] == 0))] = class_index    # Extensive
        return data


class BinaryClosingForegroundd(MapTransform):
    '''Expects single channel integer labels, numpy array format.'''
    def __init__(
        self,
        keys: KeysCollection,
        width: int = 4,
        connectivity: int = 3,
        allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.width = width
        self.connectivity = connectivity
        self.footprint = np.ones((width, width, width), dtype=bool)  # Cubic footprint
        self.keeplargest = KeepLargestConnectedComponent(connectivity=3)
        
    def __call__(self, data: Mapping[Hashable, Any]) -> dict:
        for key in self.key_iterator(data):
            orig_fg = (data[key] > 0).squeeze(0)
            closed = binary_closing(orig_fg, footprint=self.footprint)
            closed = orig_fg | closed  # force extensivity explicitly
            closed = np.expand_dims(closed, 0)
            closed = self.keeplargest(closed)
            data[key][~closed] = 0
        return data