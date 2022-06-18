from os import path
import logging
from typing import Tuple, Union

import torch
from torch import nn

from network.MFNET import MFNET_3D
from network.TorchUtils import TorchModel
from network.c3d import C3D
from network.resnet import generate_model
from utils.types import Device


def load_feature_extractor(
    features_method: str, feature_extractor_path: str, device: Union[torch.device, str]
) -> nn.Module:
    if not path.exists(feature_extractor_path):
        raise FileNotFoundError(
            f"Couldn't find feature extractor {feature_extractor_path}.\n"
            + r"If you are using resnet, download it first from:\n"
            + r"r3d101: https://drive.google.com/file/d/1kQAvOhtL-sGadblfd3NmDirXq8vYQPvf/view?usp=sharing"
            + "\n"
            + r"r3d152: https://drive.google.com/uc?id=17wdy_DS9UY37J9XTV5XCLqxOFgXiv3ZK&export=download"
        )
    logging.info(f"Loading feature extractor from {feature_extractor_path}")

    model = None
    if features_method == "c3d":
        model = C3D(pretrained=feature_extractor_path)
    elif features_method == "mfnet":
        model = MFNET_3D()
        model.load_state(state_dict=feature_extractor_path)
    elif features_method == "r3d101":
        model = generate_model(model_depth=101)
        param_dict = torch.load(feature_extractor_path)["state_dict"]
        param_dict.pop("fc.weight")
        param_dict.pop("fc.bias")
        model.load_state_dict(param_dict)
    elif features_method == "r3d101":
        model = generate_model(model_depth=152)
        param_dict = torch.load(feature_extractor_path)["state_dict"]
        param_dict.pop("fc.weight")
        param_dict.pop("fc.bias")
        model.load_state_dict(param_dict)
    else:
        raise NotImplementedError(
            f"Features extraction method {features_method} not implemented"
        )

    return model.to(device)


def load_anomaly_detector(ad_model_path: str, device: Device):
    assert path.exists(ad_model_path)
    logging.info(f"Loading anomaly detector from {ad_model_path}")

    anomaly_detector = TorchModel.load_model(ad_model_path).to(device)
    return anomaly_detector.eval()


def load_models(
    feature_extractor_path: str,
    ad_model_path: str,
    features_method: str = "c3d",
    device: str = "cuda",
) -> Tuple[nn.Module, nn.Module]:
    """
    Loads both feature extractor and anomaly detector from the given paths
    :param feature_extractor_path: path of the features extractor weights to load
    :param ad_model_path: path of the anomaly detector weights to load
    :param features_method: name of the model to use for features extraction
    :param device: device to use for the models
    :return: anomaly_detector, feature_extractor
    """
    feature_extractor = load_feature_extractor(
        features_method, feature_extractor_path, device
    )
    anomaly_detector = load_anomaly_detector(ad_model_path, device)
    return anomaly_detector, feature_extractor
