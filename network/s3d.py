"""This module contains a wrapper for the S3D video model from torchvision."""

import torch
from torch import Tensor, nn
from torchvision.models.video import S3D_Weights
from torchvision.models.video import s3d as s3d_model


class S3D(nn.Module):
    """S3D feature extractor.

    Uses separable 3D convolutions with an Inception-style architecture.
    Outputs 1024-dimensional feature vectors.

    When no pretrained path is provided, loads Kinetics-400 pretrained weights
    from torchvision.
    """

    def __init__(self, pretrained: str | None = None) -> None:
        super().__init__()
        if pretrained:
            model = s3d_model(weights=None)
            model.load_state_dict(torch.load(pretrained, weights_only=True))
        else:
            model = s3d_model(weights=S3D_Weights.DEFAULT)
        model.classifier = nn.Sequential(nn.Identity())
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
