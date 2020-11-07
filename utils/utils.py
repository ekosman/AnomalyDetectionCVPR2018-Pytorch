import logging
import sys

import torch
from torchvision.transforms import transforms

from . import transforms_video


def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def register_logger(log_file, stdout=True):
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)

    handlers = []

    if stdout:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(format="%(asctime)s %(message)s",
                        handlers=handlers,
                        level=logging.INFO,
                        )
    logging.root.setLevel(logging.INFO)


def build_transforms():
    mean = [124 / 255, 117 / 255, 104 / 255]
    std = [1 / (.0167 * 255)] * 3
    res = transforms.Compose([
        transforms_video.ToTensorVideo(),
        transforms_video.RandomResizedCropVideo(112, 112),
        transforms_video.NormalizeVideo(mean=mean, std=std)
    ])

    return res
