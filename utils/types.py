"""This module contains new types used in this library."""

from typing import Union

from torch import device

from network.c3d import C3D
from network.MFNET import MFNET_3D
from network.mvit import MViT
from network.r2plus1d import R2Plus1D
from network.resnet import ResNet
from network.s3d import S3D

Device = Union[str, device]
FeatureExtractor = Union[C3D, ResNet, MFNET_3D, R2Plus1D, S3D, MViT]
