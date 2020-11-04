import argparse

import torch
import torch.backends.cudnn as cudnn

from features_loader import FeaturesLoader
from network.anomaly_detector_model import AnomalyDetector, custom_objective, RegularizedLoss
from network.model import model
from utils import metric
from utils.lr_scheduler import MultiFactorScheduler
from utils.utils import register_logger


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")

    # io
    parser.add_argument('--features_path', default='features',
                        help="path to features")
    parser.add_argument('--annotation_path', default="Train_Annotation.txt",
                        help="path to train annotation")
    parser.add_argument('--model-dir', type=str, default="./exps/model",
                        help="set logging file.")
    parser.add_argument('--log-file', type=str, default="log.log",
                        help="set logging file.")

    # optimization
    parser.add_argument('--batch-size', type=int, default=60,
                        help="batch size")
    parser.add_argument('--lr-base', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--lr-steps', type=list, default=[int(1e4 * x) for x in [5, 10, 15]],
                        help="number of samples to pass before changing learning rate")  # 1e6 million

    parser.add_argument('--lr-factor', type=float, default=1,
                        help="reduce the learning with factor")
    parser.add_argument('--end-epoch', type=int, default=20000,
                        help="maxmium number of training epoch")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    register_logger(log_file=args.log_file)

    torch.manual_seed(args.random_seed)

    train_loader = FeaturesLoader(features_path=args.features_path,
                                  annotation_path=args.annotation_path)

    train_iter = torch.utils.data.DataLoader(train_loader,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=8,  # 4, # change this part accordingly
                                             pin_memory=True)

    network = AnomalyDetector()
    net = model(net=network,
                criterion=RegularizedLoss(network, custom_objective).cuda(),
                model_prefix=args.model_dir,
                step_callback_freq=5,
                save_checkpoint_freq=args.save_frequency,
                opt_batch_size=args.batch_size,  # optional, 60 in the paper
                )

    if torch.cuda.is_available():
        net.net.cuda()
        torch.cuda.manual_seed(args.random_seed)
        net.net = torch.nn.DataParallel(net.net).cuda()

    """
        In the original paper:
        lr = 0.01
        epsilon = 1e-8
    """
    optimizer = torch.optim.Adadelta(net.net.parameters(),
                                     lr=args.lr_base,
                                     eps=1e-8)

    # set learning rate scheduler
    lr_scheduler = MultiFactorScheduler(base_lr=args.lr_base,
                                        steps=[int(x / (args.batch_size)) for x in args.lr_steps],
                                        factor=args.lr_factor,
                                        step_counter=0)
    # define evaluation metric
    metrics = metric.MetricList(metric.Loss(name="loss-ce"), )
    # enable cudnn tune
    cudnn.benchmark = True

    net.fit(train_iter=train_iter,
            eval_iter=None,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            epoch_start=0,
            epoch_end=args.end_epoch)
