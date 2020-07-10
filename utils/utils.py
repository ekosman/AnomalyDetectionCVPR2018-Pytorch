from torchvision.transforms import transforms
from . import transforms_video
import os
import logging


def set_logger(log_file='', debug_mode=False):
    if log_file:
        if not os.path.exists("./" + os.path.dirname(log_file)):
            os.makedirs("./" + os.path.dirname(log_file))
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]

    """ add '%(filename)s:%(lineno)d %(levelname)s:' to format show source file """
    logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers)


def build_transforms():
    mean = [124 / 255, 117 / 255, 104 / 255]
    std = [1 / (.0167 * 255)] * 3
    res = transforms.Compose([
        transforms_video.ToTensorVideo(),
        transforms_video.RandomResizedCropVideo(112, 112),
        transforms_video.NormalizeVideo(mean=mean, std=std)
    ])

    return res
