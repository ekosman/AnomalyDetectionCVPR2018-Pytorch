import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from features_loader import FeaturesDatasetWrapper
from network.anomaly_detector_model import AnomalyDetector, custom_objective, RegularizedLoss
from network.model import model
from utils.callbacks import TensorboardCallback
from utils.utils import register_logger
import pytorch_wrapper as pw
from os import path
import os


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")

    # io
    parser.add_argument('--features_path', default='features',
                        help="path to features")
    parser.add_argument('--annotation_path', default="Train_Annotation.txt",
                        help="path to train annotation")
    parser.add_argument('--log-file', type=str, default="log.log",
                        help="set logging file.")
    parser.add_argument('--exps_dir', type=str, default="exps",
                        help="set logging file.")
    parser.add_argument('--save_name', type=str, default="model",
                        help="name of the saved model.")

    # optimization
    parser.add_argument('--batch-size', type=int, default=60,
                        help="batch size")
    parser.add_argument('--lr-base', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--end-epoch', type=int, default=20000,
                        help="maxmium number of training epoch")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    register_logger(log_file=args.log_file)
    os.makedirs(args.exps_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True  # enable cudnn tune

    train_loader = FeaturesDatasetWrapper(features_path=args.features_path, annotation_path=args.annotation_path)

    train_iter = torch.utils.data.DataLoader(train_loader,
                                             batch_size=args.batch_size,
                                             num_workers=0,  # 4, # change this part accordingly
                                             pin_memory=True)

    network = AnomalyDetector()
    system = pw.System(model=network, device=device)

    """
    In the original paper:
        lr = 0.01
        epsilon = 1e-8
    """
    optimizer = torch.optim.Adadelta(network.parameters(), lr=args.lr_base, eps=1e-8)

    loss_wrapper = pw.loss_wrappers.GenericPointWiseLossWrapper(RegularizedLoss(network, custom_objective))

    tb_writer = SummaryWriter(log_dir=args.exps_dir)

    system.train(
        loss_wrapper,
        optimizer,
        train_data_loader=train_iter,
        callbacks=[pw.training_callbacks.NumberOfEpochsStoppingCriterionCallback(args.end_epoch),
                   TensorboardCallback(tb_writer)]
    )

    system.save_model_state(path.join(args.exps_dir, f'{args.save_name}.weights'))
