from typing import Hashable, Any, Mapping, Sequence, Union

import numpy as np
import torch
from skimage.morphology import binary_closing

from monai.config import KeysCollection
from monai.transforms import MapTransform, KeepLargestConnectedComponent
from skimage.morphology import binary_closing, ball

class BinaryClosingd(MapTransform):
    """
    Expects single channel integer labels
    """

    def __init__(
        self,
        keys: "KeysCollection",
        radius: int = 4,
        classes: int = 2,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if classes is not None and (not isinstance(classes, int) or classes < 1):
            raise ValueError("classes must be a positive integer or None.")
        self.radius = radius
        self.classes = classes
        self.footprint = ball(radius)

    def __call__(self, data: Mapping[Hashable, Any]) -> dict:
        for key in self.key_iterator(data):
            for class_index in range(1, self.classes):
                fg = (data[key] == class_index).squeeze(0)
                fg = binary_closing(fg, footprint=self.footprint).unsqueeze(0).bool()
                data[key][(fg & (data[key] == 0))] = class_index
                data[key][(~fg & (data[key] == class_index))] = 0
        return data


class BinaryClosingForegroundd(MapTransform):
    '''Expects integer labels'''
    def __init__(
        self,
        keys: KeysCollection,
        dilation: int = 8,
        connectivity: int = 1,
        allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)

        self.dilation = dilation
        self.connectivity = connectivity
        self.footprint = ball(self.dilation)
        self.keeplargest = KeepLargestConnectedComponent(connectivity=3)
        
    def __call__(self, data: Mapping[Hashable, Any]) -> dict:
        for key in self.key_iterator(data):
            fg = (data[key] > 0).squeeze(0).bool()  # assume single channel
            fg = binary_closing(fg, footprint=self.footprint).unsqueeze(0)
            fg = self.keeplargest(fg).bool()
            data[key][~fg] = 0
        return data