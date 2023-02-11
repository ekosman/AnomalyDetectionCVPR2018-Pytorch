""""This module contains a video feature loader."""

import logging
import os
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils import data

from feature_extractor import read_features


class FeaturesLoader:
    """Loads video features that are stored as text files."""

    def __init__(
        self,
        features_path: str,
        feature_dim: int,
        annotation_path: str,
        bucket_size: int = 30,
        iterations: int = 20000,
    ) -> None:
        """
        Args:
            features_path: Path to the directory that contains the features in text files
            feature_dim: Dimensionality of each feature vector
            annotation_path: Path to the annotation file
            bucket_size: Size of each bucket
            iterations: How many iterations the loader should perform
        """

        super().__init__()
        self._features_path = features_path
        self._feature_dim = feature_dim
        self._bucket_size = bucket_size

        # load video list
        (
            self.features_list_normal,
            self.features_list_anomaly,
        ) = FeaturesLoader._get_features_list(
            features_path=self._features_path, annotation_path=annotation_path
        )

        self._iterations = iterations
        self._features_cache = {}
        self._i = 0

    def __len__(self) -> int:
        return self._iterations

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if self._i == len(self):
            self._i = 0
            raise StopIteration

        succ = False
        while not succ:
            try:
                feature, label = self.get_features()
                succ = True
            except Exception as e:
                index = np.random.choice(range(0, self.__len__()))
                logging.warning(
                    f"VideoIter:: ERROR!! (Force using another index:\n{index})\n{e}"
                )

        self._i += 1
        return feature, label

    def get_features(self) -> Tuple[Tensor, Tensor]:
        """Fetches a bucket sample from the dataset."""
        normal_paths = np.random.choice(
            self.features_list_normal, size=self._bucket_size
        )
        abnormal_paths = np.random.choice(
            self.features_list_anomaly, size=self._bucket_size
        )
        all_paths = np.concatenate([normal_paths, abnormal_paths])
        features = torch.stack(
            [
                read_features(f"{feature_subpath}.txt", self._features_cache)
                for feature_subpath in all_paths
            ]
        )
        return (
            features,
            torch.cat([torch.zeros(self._bucket_size), torch.ones(self._bucket_size)]),
        )

    @staticmethod
    def _get_features_list(
        features_path: str, annotation_path: str
    ) -> Tuple[List[str], List[str]]:
        """Retrieves the paths of all feature files contained within the
        annotation file.

        Args:
            features_path: Path to the directory that contains feature text files
            annotation_path: Path to the annotation file

        Returns:
            Tuple[List[str], List[str]]: Two list that contain the corresponding paths of normal and abnormal
                feature files.
        """
        assert os.path.exists(features_path)
        features_list_normal = []
        features_list_anomaly = []
        with open(annotation_path) as f:
            lines = f.read().splitlines(keepends=False)
            for line in lines:
                items = line.split()
                file = items[0].split(".")[0]
                file = file.replace("/", os.sep)
                feature_path = os.path.join(features_path, file)
                if "Normal" in feature_path:
                    features_list_normal.append(feature_path)
                else:
                    features_list_anomaly.append(feature_path)

        return features_list_normal, features_list_anomaly


class FeaturesLoaderVal(data.Dataset):
    """Loader for video features for validation phase."""

    def __init__(self, features_path, annotation_path):
        super().__init__()
        self.features_path = features_path
        # load video list
        self.state = "Normal"
        self.features_list = FeaturesLoaderVal._get_features_list(
            features_path=features_path, annotation_path=annotation_path
        )

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, index: int):
        succ = False
        while not succ:
            try:
                data = self.get_feature(index)
                succ = True
            except Exception as e:
                logging.warning(
                    f"VideoIter:: ERROR!! (Force using another index:\n{index})\n{e}"
                )

        return data

    def get_feature(self, index: int):
        """Fetch feature that matches given index in the dataset.

        Args:
            index (int): Index of the feature to fetch.

        Returns:
            _type_: _description_
        """
        feature_subpath, start_end_couples, length = self.features_list[index]
        features = read_features(f"{feature_subpath}.txt")
        return features, start_end_couples, length

    @staticmethod
    def _get_features_list(
        features_path: str, annotation_path: str
    ) -> List[Tuple[str, List[List[int]], int]]:
        """Retrieves the paths of all feature files contained within the
        annotation file.

        Args:
            features_path: Path to the directory that contains feature text files
            annotation_path: Path to the annotation file

        Returns:
            List[Tuple[str, Tensor, int]]: A list of tuples that describe each video and the temporal annotations
                of anomalies in the videos
        """
        assert os.path.exists(features_path)
        features_list = []
        with open(annotation_path) as f:
            lines = f.read().splitlines(keepends=False)
            for line in lines:
                start_end_couples = []
                items = line.split()
                anomalies_frames = [int(x) for x in items[3:]]
                start_end_couples = torch.tensor(
                    [anomalies_frames[:2], anomalies_frames[2:]]
                )
                file = items[0].split(".")[0]
                file = file.replace("/", os.sep)
                feature_path = os.path.join(features_path, file)
                length = int(items[1])

                features_list.append((feature_path, start_end_couples, length))

        return features_list
