"""This module contains a wrapper for the MViT V2 video model from torchvision."""

import torch
from torch import Tensor, nn
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s


class MViT(nn.Module):
    """MViT V2 Small feature extractor.

    Uses a multiscale vision transformer architecture for video understanding.
    Outputs 768-dimensional feature vectors.

    When no pretrained path is provided, loads Kinetics-400 pretrained weights
    from torchvision.
    """

    def __init__(self, pretrained: str | None = None) -> None:
        super().__init__()
        if pretrained:
            model = mvit_v2_s(weights=None)
            model.load_state_dict(torch.load(pretrained, weights_only=True))
        else:
            model = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)
        model.head = nn.Identity()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
