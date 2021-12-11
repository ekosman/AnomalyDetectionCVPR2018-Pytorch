import logging
import sys

import torch
from torchvision.transforms import transforms

from . import transforms_video


def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def register_logger(log_file=None, stdout=True):
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)

    handlers = []

    if stdout:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(format="%(asctime)s %(message)s",
                        handlers=handlers,
                        level=logging.INFO,
                        )
    logging.root.setLevel(logging.INFO)


def build_transforms(mode='c3d'):
    if mode == 'c3d':
        mean = [124 / 255, 117 / 255, 104 / 255]
        std = [1 / (.0167 * 255)] * 3
        resize = 128, 171
        crop = 112
    elif mode == 'i3d':
        mean = [0, 0, 0]
        std = [1, 1, 1]
        size = 224
    elif mode == 'mfnet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        size = 224
    elif mode == '3dResNet':
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
        size = 224
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")
    
    if mode == 'c3d':
        res = transforms.Compose([
            transforms_video.ToTensorVideo(),
            transforms_video.ResizeVideo(resize),
            transforms_video.CenterCropVideo(crop),
            transforms_video.NormalizeVideo(mean=mean, std=std)
        ])
    else:
        res = transforms.Compose([
            transforms_video.ToTensorVideo(),
            transforms_video.NormalizeVideo(mean=mean, std=std)
        ])

    return res
