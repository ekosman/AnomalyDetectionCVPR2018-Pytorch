import argparse
import logging
import os
from os import path, mkdir

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import video_transforms as transforms
from c3d import C3D
from data_loader import VideoIterTrain

parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
# debug
parser.add_argument('--debug-mode', type=bool, default=True,
                    help="print all setting for debugging.")
# io
parser.add_argument('--dataset_path', default='../kinetics2/kinetics2/AnomalyDetection',
                    help="path to dataset")
parser.add_argument('--annotation_path', default="Train_Annotation.txt",
                    help="path to train annotation")
parser.add_argument('--annotation_path_test', default="Test_Annotation.txt",
                    help="path to test annotation")
parser.add_argument('--clip-length', type=int, default=16,
                    help="define the length of each input sample.")
parser.add_argument('--train-frame-interval', type=int, default=2,
                    help="define the sampling interval between frames.")
parser.add_argument('--val-frame-interval', type=int, default=2,
                    help="define the sampling interval between frames.")
parser.add_argument('--task-name', type=str, default='',
                    help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="./exps/models",
                    help="set logging file.")
parser.add_argument('--log-file', type=str, default="",
                    help="set logging file.")

# device
parser.add_argument('--gpus', type=str, default="0,1,2,3",
                    help="define gpu id")
parser.add_argument('--pretrained_3d', type=str,
                    default='',
                    help="load default 3D pretrained model.")
parser.add_argument('--resume-epoch', type=int, default=-1,
                    help="resume train")
# optimization
parser.add_argument('--fine-tune', type=bool, default=True, help="apply different learning rate for different layers")
parser.add_argument('--batch-size', type=int, default=8,
                    help="batch size")
parser.add_argument('--lr-base', type=float, default=0.005,
                    help="learning rate")
parser.add_argument('--lr_mult_old_layers', type=float, default=0.2,
                    help="learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help="weight_decay")
parser.add_argument('--lr-steps', type=list, default=[int(1e4 * x) for x in [5, 10, 15]],
                    ##In spli2 we can make the step a bit later
                    help="number of samples to pass before changing learning rate")  # 1e6 million

parser.add_argument('--lr-factor', type=float, default=0.1,
                    help="reduce the learning with factor")
parser.add_argument('--save-frequency', type=float, default=1,
                    help="save once after N epochs")
parser.add_argument('--end-epoch', type=int, default=100,
                    help="maxmium number of training epoch")
parser.add_argument('--random-seed', type=int, default=1,
                    help='random seed (default: 1)')

features_dir = r"features2"

current_path = None
current_dir = None
current_data = None


class FeaturesWriter:
    def __init__(self, chunk_size=16):
        self.path = None
        self.dir = None
        self.data = None
        self.chunk_size = chunk_size

    def _init_video(self, video_name, dir):
        self.path = path.join(dir, f"{video_name}.pickle")
        self.dir = dir
        self.data = dict()

    def has_video(self):
        return self.data is not None

    def dump(self):
        print(f'Dumping {self.path}')
        if not path.exists(self.dir):
            os.mkdir(self.dir)

        features = np.array([self.data[key] for key in sorted(self.data)])
        features = features / np.expand_dims(np.linalg.norm(features, ord=2, axis=-1), axis=-1)
        padding_count = int(32 * np.ceil(features.shape[0] / 32) - features.shape[0])
        features = torch.from_numpy(np.vstack([features, torch.zeros(padding_count, 4096)]))
        segments = torch.stack(torch.chunk(features, chunks=32, dim=0))
        avg_segments = segments.mean(dim=-2).numpy()
        with open(self.path, 'w') as fp:
            for d in avg_segments:
                d = [str(x) for x in d]
                fp.write(' '.join(d) + '\n')

    def _is_new_video(self, video_name, dir):
        new_path = path.join(dir, f"{video_name}.pickle")
        if self.path != new_path and self.path is not None:
            return True

        return False

    def store(self, feature, start_frame):
        self.data[start_frame // self.chunk_size] = list(feature.cpu().numpy())

    def write(self, feature, video_name, start_frame, dir):
        if not self.has_video():
            self._init_video(video_name, dir)

        if self._is_new_video(video_name, dir):
            self.dump()
            self._init_video(video_name, dir)

        self.store(feature, start_frame)


def read_features(video_name, dir):
    file_path = f"{video_name}.txt"
    file_path = path.join(dir, file_path)
    if not path.exists(file_path):
        raise Exception(f"Feature doesn't exist: {file_path}")
    features = None
    with open(file_path, 'r') as fp:
        data = fp.read().splitlines(keepends=False)
        features = np.zeros((len(data), 4096))
        for i, line in enumerate(data):
            features[i, :] = [float(x) for x in line.split(' ')]

    return torch.from_numpy(features)


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    args = parser.parse_args()
    set_logger(log_file=args.log_file, debug_mode=args.debug_mode)

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    cudnn.benchmark = True

    mean = [124 / 255, 117 / 255, 104 / 255]
    std = [1 / (.0167 * 255)] * 3
    normalize = transforms.Normalize(mean=mean, std=std)

    train_loader = VideoIterTrain(dataset_path=args.dataset_path,
                                  annotation_path=args.annotation_path,
                                  clip_length=args.clip_length,
                                  frame_stride=args.train_frame_interval,
                                  video_transform=transforms.Compose([
                                      transforms.Resize((256, 256)),
                                      transforms.RandomCrop((224, 224)),
                                      transforms.ToTensor(),
                                      normalize,
                                  ]),
                                  name='train',
                                  return_item_subpath=False,
                                  )

    train_iter = torch.utils.data.DataLoader(train_loader,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=32,  # 4, # change this part accordingly
                                             pin_memory=True)

    val_loader = VideoIterTrain(dataset_path=args.dataset_path,
                                annotation_path=args.annotation_path_test,
                                clip_length=args.clip_length,
                                frame_stride=args.val_frame_interval,
                                video_transform=transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.RandomCrop((224, 224)),
                                    transforms.ToTensor(),
                                    normalize,
                                ]),
                                name='val',
                                return_item_subpath=False,
                                )

    val_iter = torch.utils.data.DataLoader(val_loader,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=32,  # 4, # change this part accordingly
                                           pin_memory=True)

    network = C3D(pretrained=args.pretrained_3d)
    network.to(device)

    if not path.exists(features_dir):
        mkdir(features_dir)

    features_writer = FeaturesWriter()

    for i_batch, (data, target, sampled_idx, dirs, vid_names) in tqdm(enumerate(train_iter)):
        with torch.no_grad():
            outputs = network(data.cuda())

            for i, (dir, vid_name, start_frame) in enumerate(zip(dirs, vid_names, sampled_idx.cpu().numpy())):
                dir = path.join(features_dir, dir)
                features_writer.write(feature=outputs[i], video_name=vid_name, start_frame=start_frame, dir=dir)

    features_writer.dump()

    features_writer = FeaturesWriter()
    for i_batch, (data, target, sampled_idx, dirs, vid_names) in tqdm(enumerate(val_iter)):
        with torch.no_grad():
            outputs = network(data.cuda())

            for i, (dir, vid_name, start_frame) in enumerate(zip(dirs, vid_names, sampled_idx.cpu().numpy())):
                dir = path.join(features_dir, dir)
                features_writer.write(feature=outputs[i], video_name=vid_name, start_frame=start_frame, dir=dir)

    features_writer.dump()


if __name__ == "__main__":
    main()
