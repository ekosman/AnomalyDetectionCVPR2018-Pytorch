"""This module contains a training procedure for video feature extraction."""

import argparse
import logging
import os
from os import mkdir, path

import numpy as np
import torch
from torch import Tensor
from torch.backends import cudnn
from torch.utils.data import DataLoader

from data_loader import VideoIter
from network.TorchUtils import get_torch_device
from utils.load_model import load_feature_extractor
from utils.utils import build_transforms, register_logger


def get_args() -> argparse.Namespace:
    """Reads command line args and returns the parser object the represent the
    specified arguments."""

    parser = argparse.ArgumentParser(description="Video Feature Extraction Parser")

    # io
    parser.add_argument(
        "--dataset_path",
        default="../kinetics2/kinetics2/AnomalyDetection",
        help="path to dataset",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=16,
        help="define the length of each input sample.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="define the number of workers used for loading the videos",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="define the sampling interval between frames.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="log the writing of clips every n steps.",
    )
    parser.add_argument("--log-file", type=str, help="set logging file.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="features",
        help="set output directory for the features.",
    )

    # optimization
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")

    # model
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="type of feature extractor",
        choices=["c3d", "i3d", "mfnet", "3dResNet", "r2plus1d", "s3d", "mvit"],
    )
    parser.add_argument(
        "--pretrained_3d", type=str, help="load default 3D pretrained model."
    )

    return parser.parse_args()


def to_segments(data: Tensor | np.ndarray, n_segments: int = 32) -> list[np.ndarray]:
    """These code is taken from:

        # https://github.com/rajanjitenpatel/C3D_feature_extraction/blob/b5894fa06d43aa62b3b64e85b07feb0853e7011a/extract_C3D_feature.py#L805

    Args:
        data (Union[Tensor, np.ndarray]): List of features of a certain video
        n_segments (int, optional): Number of segments

    Returns:
        List[np.ndarray]: List of `num` segments
    """
    data = np.array(data)
    Segments_Features = []
    thirty2_shots = np.round(np.linspace(0, len(data) - 1, num=n_segments + 1)).astype(
        int
    )
    for ss, ee in zip(thirty2_shots[:-1], thirty2_shots[1:]):
        if ss == ee:
            temp_vect = data[min(ss, data.shape[0] - 1), :]
        else:
            temp_vect = data[ss:ee, :].mean(axis=0)

        temp_vect = temp_vect / np.linalg.norm(temp_vect)

        if np.linalg.norm(temp_vect) != 0:
            Segments_Features.append(temp_vect.tolist())

    return Segments_Features


class FeaturesWriter:
    """Accumulates and saves extracted features."""

    def __init__(self, num_videos: int, chunk_size: int = 16) -> None:
        self.path = ""
        self.dir = ""
        self.data = {}
        self.chunk_size = chunk_size
        self.num_videos = num_videos
        self.dump_count = 0

    def _init_video(self, video_name: str, dir: str) -> None:
        """Initialize the state of the writer for a new video.

        Args:
            video_name (str): Name of the video to initialize for.
            dir (str): Directory where the video is stored.
        """
        self.path = path.join(dir, f"{video_name}.txt")
        self.dir = dir
        self.data = {}

    def has_video(self) -> bool:
        """Checks whether the writer is initialized with a video.

        Returns:
            bool
        """
        return self.data is not None

    def dump(self, dir: str) -> None:
        """Saves the accumulated features to disk.

        The features will be segmented and normalized.
        """
        logging.info(f"{self.dump_count} / {self.num_videos}:	Dumping {self.path}")
        self.dump_count += 1
        self.dir = dir
        if not path.exists(self.dir):
            os.makedirs(self.dir, exist_ok=True)
        #####################################################
        # Check if data is empty before attempting to process it
        if len(self.data) == 0:
            logging.warning("No data to dump, skipping.")
            return  # If data is empty, skip this dump.
        #####################################################
        features = to_segments(np.array([self.data[key] for key in sorted(self.data)]))
        with open(self.path, "w") as fp:
            for d in features:
                d_str = [str(x) for x in d]
                fp.write(" ".join(d_str) + "\n")

    def _is_new_video(self, video_name: str, dir: str) -> bool:
        """Checks whether the given video is new or the writer is already
        initialized with it.

        Args:
            video_name (str): Name of the possibly new video.
            dir (str): Directory where the video is stored.

        Returns:
            bool
        """
        new_path = path.join(dir, f"{video_name}.txt")
        if self.path != new_path and self.path is not None:
            return True

        return False

    def store(self, feature: Tensor | np.ndarray, idx: int) -> None:
        """Accumulate features.

        Args:
            feature (Union[Tensor, np.ndarray]): Features to be accumulated.
            idx (int): Indices of features in the video.
        """
        self.data[idx] = list(feature)

    def write(
        self, feature: Tensor | np.ndarray, video_name: str, idx: int, dir: str
    ) -> None:
        if not self.has_video():
            self._init_video(video_name, dir)
        if self._is_new_video(video_name, dir):
            self.dump(dir)
            self._init_video(video_name, dir)

        self.store(feature, idx)


def read_features(file_path, cache: dict[str, Tensor] | None = None) -> Tensor:
    """Reads features from file.

    Args:
        file_path (_type_): Path to a text file containing features. Each line should contain a feature
            for a single video segment.
        cache (Dict, optional): A cache that stores features that were already loaded.
            If `None`, caching is disabled.Defaults to None.

    Raises:
        FileNotFoundError: The provided path does not exist.

    Returns:
        Tensor
    """
    if cache is not None and file_path in cache:
        return cache[file_path]

    if not path.exists(file_path):
        raise FileNotFoundError(f"Feature doesn't exist: `{file_path}`")

    features = None
    with open(file_path) as fp:
        data = fp.read().splitlines(keepends=False)
        features = torch.tensor(
            np.stack([line.split(" ") for line in data]).astype(np.float32)
        )

    if cache is not None:
        cache[file_path] = features
    return features


def get_features_loader(
    dataset_path: str,
    clip_length: int,
    frame_interval: int,
    batch_size: int,
    num_workers: int,
    mode: str,
) -> tuple[VideoIter, DataLoader]:
    data_loader = VideoIter(
        dataset_path=dataset_path,
        clip_length=clip_length,
        frame_stride=frame_interval,
        video_transform=build_transforms(mode),
        return_label=False,
    )

    data_iter = torch.utils.data.DataLoader(
        data_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return data_loader, data_iter


if __name__ == "__main__":
    device = get_torch_device()

    args = get_args()
    register_logger(log_file=args.log_file)

    cudnn.benchmark = True

    data_loader, data_iter = get_features_loader(
        args.dataset_path,
        args.clip_length,
        args.frame_interval,
        args.batch_size,
        args.num_workers,
        args.model_type,
    )

    network = load_feature_extractor(args.model_type, args.pretrained_3d, device).eval()

    if not path.exists(args.save_dir):
        mkdir(args.save_dir)

    features_writer = FeaturesWriter(num_videos=data_loader.video_count)
    loop_i = 0
    global_dir: str = "none"
    with torch.no_grad():
        for data, clip_idxs, dirs, vid_names in data_iter:
            outputs = network(data.to(device)).detach().cpu().numpy()

            for i, (_dir, vid_name, clip_idx) in enumerate(
                zip(dirs, vid_names, clip_idxs)
            ):
                if loop_i == 0:
                    # pylint: disable=line-too-long
                    logging.info(
                        f"Video {features_writer.dump_count} / {features_writer.num_videos} : Writing clip {clip_idx} of video {vid_name}"
                    )
                loop_i += 1
                loop_i %= args.log_every
                _dir = path.join(args.save_dir, _dir)
                global_dir = _dir
                features_writer.write(
                    feature=outputs[i],
                    video_name=vid_name,
                    idx=clip_idx,
                    dir=_dir,
                )

    features_writer.dump(global_dir)
