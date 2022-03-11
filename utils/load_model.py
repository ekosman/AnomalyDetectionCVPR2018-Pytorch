from os import path
import logging

import torch

from network.MFNET import MFNET_3D
from network.TorchUtils import TorchModel
from network.c3d import C3D
from network.i3d import InceptionI3d
from network.resnet import generate_model


def load_feature_extractor(features_method, feature_extractor_path, device):
    assert path.exists(feature_extractor_path)
    logging.info(f"Loading feature extractor from {feature_extractor_path}")

    model = None
    if features_method == "c3d":
        model = C3D(pretrained=feature_extractor_path)
    elif features_method == "mfnet":
        model = MFNET_3D()
        model.load_state(state_dict=feature_extractor_path)
    elif features_method == "3dResNet":
        model = generate_model(model_depth=101)
        param_dict = torch.load(feature_extractor_path)["state_dict"]
        param_dict.pop("fc.weight")
        param_dict.pop("fc.bias")
        model.load_state_dict(param_dict)
    else:
        raise NotImplementedError(
            f"Features extraction method {features_method} not implemented"
        )

    return model.to(device)


def load_anomaly_detector(ad_model_path, device):
    assert path.exists(ad_model_path)
    logging.info(f"Loading anomaly detector from {ad_model_path}")

    anomaly_detector = TorchModel.load_model(ad_model_path).to(device)
    return anomaly_detector.eval()


def load_models(
    feature_extractor_path, ad_model_path, features_method="c3d", device="cuda"
):
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
