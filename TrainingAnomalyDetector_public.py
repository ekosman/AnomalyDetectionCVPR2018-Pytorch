""" "This module contains a training procedure for video anomaly detection."""

import argparse
from os import makedirs, path

import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from features_loader import FeaturesLoader
from network.anomaly_detector_model import (
    AnomalyDetector,
    RegularizedLoss,
    custom_objective,
)
from network.TorchUtils import TorchModel, get_torch_device
from utils.callbacks import DefaultModelCallback, TensorBoardCallback
from utils.utils import register_logger


def get_args() -> argparse.Namespace:
    """Reads command line args and returns the parser object the represent the
    specified arguments."""
    parser = argparse.ArgumentParser(
        description="Video Anomaly Detection Training Parser"
    )

    # io
    parser.add_argument("--features_path", default="features", help="path to features")
    parser.add_argument(
        "--annotation_path",
        default="Train_Annotation.txt",
        help="path to train annotation",
    )
    parser.add_argument(
        "--log_file", type=str, default="log.log", help="set logging file."
    )
    parser.add_argument(
        "--exps_dir",
        type=str,
        default="exps",
        help="path to the directory where models and tensorboard would be saved.",
    )
    parser.add_argument(
        "--checkpoint", type=str, help="load a model for resume training"
    )

    # optimization
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="epochs interval for saving the model checkpoints",
    )
    parser.add_argument("--lr_base", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--iterations_per_epoch",
        type=int,
        default=20000,
        help="number of training iterations",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="number of training epochs"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Register directories
    register_logger(log_file=args.log_file)
    makedirs(args.exps_dir, exist_ok=True)
    models_dir = path.join(args.exps_dir, "models")
    tb_dir = path.join(args.exps_dir, "tensorboard")
    makedirs(models_dir, exist_ok=True)
    makedirs(tb_dir, exist_ok=True)

    # Optimizations
    device = get_torch_device()
    cudnn.benchmark = True  # enable cudnn tune

    # Data loader
    train_loader = FeaturesLoader(
        features_path=args.features_path,
        annotation_path=args.annotation_path,
        iterations=args.iterations_per_epoch,
    )

    feature_dim = train_loader.get_feature_dim

    # Model
    if args.checkpoint is not None and path.exists(args.checkpoint):
        model = TorchModel.load_model(args.checkpoint)
        assert (
            feature_dim == model.model.input_dim
        ), f"Dimentionality mismatch between input of the model ({model.input_dim}) and the loader ({feature_dim})"
    else:
        network = AnomalyDetector(feature_dim)
        model = TorchModel(network)

    model = model.to(device).train()
    # Training parameters
    """
    In the original paper:
        lr = 0.01
        epsilon = 1e-8
    """
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr_base, eps=1e-8)

    criterion = RegularizedLoss(model.get_model(), custom_objective).to(device)

    # Callbacks
    tb_writer = SummaryWriter(log_dir=tb_dir)
    model.register_callback(DefaultModelCallback(visualization_dir=args.exps_dir))
    model.register_callback(TensorBoardCallback(tb_writer=tb_writer))

    # Training
    model.fit(
        train_iter=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        network_model_path_base=models_dir,
        save_every=args.save_every,
    )
