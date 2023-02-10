"""This module contains utilities for anomaly detection."""
import logging
import sys
from typing import List, Optional, Union

from torchvision.transforms import transforms

from . import transforms_video


def register_logger(log_file: Optional[str] = None, stdout: bool = True) -> None:
    """Register a logger.

    Args:
        log_file (str, optional): Path to the file where log should be written.
            If `None`, log wouldn't be written to any file. Defaults to None.
        stdout (bool, optional): If `True`, the log would be printed to stdout. Defaults to True.
    """
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)

    handlers: List[Union[logging.FileHandler, logging.StreamHandler]] = []

    if stdout:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="%(asctime)s %(message)s",
        handlers=handlers,
        level=logging.INFO,
    )
    logging.root.setLevel(logging.INFO)


def build_transforms(mode: str = "c3d") -> transforms.Compose:
    """Build transforms to use for training an anomaly detection model.

    Args:
        mode (str, optional): Mode for which transforms should be constructed.
        Either c3d | i3d | mfnet | 3dResNet. Defaults to "c3d".

    Raises:
        NotImplementedError: The provided mode is not implemented.

    Returns:
        transforms.Compose
    """
    if mode == "c3d":
        mean = [124 / 255, 117 / 255, 104 / 255]
        std = [1 / (0.0167 * 255)] * 3
        resize = 128, 171
        crop = 112
    elif mode == "i3d":
        mean = [0, 0, 0]
        std = [1, 1, 1]
    elif mode == "mfnet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif mode == "3dResNet":
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")

    if mode == "c3d":
        res = transforms.Compose(
            [
                transforms_video.ToTensorVideo(),
                transforms_video.ResizeVideo(resize),
                transforms_video.CenterCropVideo(crop),
                transforms_video.NormalizeVideo(mean=mean, std=std),
            ]
        )
    else:
        res = transforms.Compose(
            [
                transforms_video.ToTensorVideo(),
                transforms_video.NormalizeVideo(mean=mean, std=std),
            ]
        )

    return res
