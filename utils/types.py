"""This module contains new types used in this library."""
from typing import Union

from torch import device

from network.c3d import C3D
from network.MFNET import MFNET_3D
from network.resnet import ResNet

Device = Union[str, device]
FeatureExtractor = Union[C3D, ResNet, MFNET_3D]
