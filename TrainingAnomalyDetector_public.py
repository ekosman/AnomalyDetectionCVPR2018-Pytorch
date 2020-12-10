import argparse

import pytorch_wrapper as pw
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from features_loader import FeaturesDatasetWrapper
from network.TorchUtils import TorchModel
from network.anomaly_detector_model import AnomalyDetector, custom_objective, RegularizedLoss
from utils.callbacks import TensorboardCallback, SaveCallback
from utils.utils import register_logger


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")

    # io
    parser.add_argument('--features_path', default='features',
                        help="path to features")
    parser.add_argument('--annotation_path', default="Train_Annotation.txt",
                        help="path to train annotation")
    parser.add_argument('--log_file', type=str, default="log.log",
                        help="set logging file.")
    parser.add_argument('--exps_dir', type=str, default="exps",
                        help="set logging file.")
    parser.add_argument('--save_name', type=str, default="model",
                        help="name of the saved model.")
    parser.add_argument('--checkpoint', type=str,
                        help="load a model for resume training")

    # optimization
    parser.add_argument('--batch_size', type=int, default=60,
                        help="batch size")
    parser.add_argument('--feature_dim', type=int, default=4096,
                        help="batch size")
    parser.add_argument('--save_every', type=int, default=100,
                        help="epochs interval for saving the model checkpoints")
    parser.add_argument('--lr_base', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--end_epoch', type=int, default=20000,
                        help="maxmium number of training epoch")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    register_logger(log_file=args.log_file)
    os.makedirs(args.exps_dir, exist_ok=True)
    device = get_torch_device()

    models_dir = path.join(args.exps_dir, 'models')
    tb_dir = path.join(args.exps_dir, 'tensorboard')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True  # enable cudnn tune

    train_loader = FeaturesDatasetWrapper(features_path=args.features_path, annotation_path=args.annotation_path)

    train_iter = torch.utils.data.DataLoader(train_loader,
                                             batch_size=args.batch_size,
                                             num_workers=0,
                                             pin_memory=True)

    network = AnomalyDetector(args.feature_dim)

    if args.checkpoint is not None:
        TorchModel.load_model(args.checkpoint)
    else:
        model = TorchModel(network)

    """
    In the original paper:
        lr = 0.01
        epsilon = 1e-8
    """
    optimizer = torch.optim.Adadelta(network.parameters(), lr=args.lr_base, eps=1e-8)

    criterion = RegularizedLoss(network, custom_objective)

    tb_writer = SummaryWriter(log_dir=tb_dir)

    model.register_callback()

    model.fit(train_iter=train_iter,
              criterion=criterion,
              optimizer=optimizer,
              epochs=args.epochs,
              network_model_path_base=args.models_dir,
              save_every=20000)
