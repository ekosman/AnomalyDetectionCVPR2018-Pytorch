"""This module contains a wrapper for the R(2+1)D video model from torchvision."""

import torch
from torch import Tensor, nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


class R2Plus1D(nn.Module):
    """R(2+1)D-18 feature extractor.

    Uses decomposed spatiotemporal convolutions (2D spatial + 1D temporal).
    Outputs 512-dimensional feature vectors.

    When no pretrained path is provided, loads Kinetics-400 pretrained weights
    from torchvision.
    """

    def __init__(self, pretrained: str | None = None) -> None:
        super().__init__()
        if pretrained:
            model = r2plus1d_18(weights=None)
            model.load_state_dict(torch.load(pretrained, weights_only=True))
        else:
            model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        model.fc = nn.Identity()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
