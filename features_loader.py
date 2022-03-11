import logging
import os
from os import path
from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.utils import data
from feature_extractor import read_features


class FeaturesLoader:
    def __init__(
        self,
        features_path,
        feature_dim,
        annotation_path,
        bucket_size=30,
        iterations=20000,
    ) -> None:

        super(FeaturesLoader, self).__init__()
        self.features_path = features_path
        self.feature_dim = feature_dim
        self.bucket_size = bucket_size
        # load video list
        (
            self.features_list_normal,
            self.features_list_anomaly,
        ) = FeaturesLoader._get_features_list(
            features_path=self.features_path, annotation_path=annotation_path
        )

        self.iterations = iterations
        self.features_cache = dict()
        self.i = 0

    def shuffle(self) -> None:
        self.features_list_anomaly = np.random.permutation(self.features_list_anomaly)
        self.features_list_normal = np.random.permutation(self.features_list_normal)

    def __len__(self) -> int:
        return self.iterations

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        if self.i == len(self):
            self.i = 0
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

        self.i += 1
        return feature, label

    def get_existing_features(self):
        # TODO: check if used
        res = []
        for dir in os.listdir(self.features_path):
            dir = path.join(self.features_path, dir)
            if path.isdir(dir):
                for file in os.listdir(dir):
                    file_no_ext = file.split(".")[0]
                    res.append(path.join(dir, file_no_ext))
        return res

    def get_features(self) -> Tensor:
        normal_paths = np.random.choice(
            self.features_list_normal, size=self.bucket_size
        )
        abnormal_paths = np.random.choice(
            self.features_list_anomaly, size=self.bucket_size
        )
        all_paths = np.concatenate([normal_paths, abnormal_paths])
        features = torch.stack(
            [
                read_features(
                    f"{feature_subpath}.txt", self.feature_dim, self.features_cache
                )
                for feature_subpath in all_paths
            ]
        )
        return (
            features,
            torch.cat([torch.zeros(self.bucket_size), torch.ones(self.bucket_size)]),
        )

    @staticmethod
    def _get_features_list(
        features_path: str, annotation_path: str
    ) -> Tuple[List[str], List[str]]:
        assert os.path.exists(features_path)
        features_list_normal = []
        features_list_anomaly = []
        with open(annotation_path, "r") as f:
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
    def __init__(self, features_path, feature_dim, annotation_path):

        super(FeaturesLoaderVal, self).__init__()
        self.features_path = features_path
        self.feature_dim = feature_dim
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
        feature_subpath, start_end_couples, length = self.features_list[index]
        features = read_features(f"{feature_subpath}.txt", self.feature_dim)
        return features, start_end_couples, length

    @staticmethod
    def _get_features_list(features_path: str, annotation_path: str):
        assert os.path.exists(features_path)
        features_list = []
        with open(annotation_path, "r") as f:
            lines = f.read().splitlines(keepends=False)
            for line in lines:
                start_end_couples = []
                items = line.split()
                anomalies_frames = [int(x) for x in items[3:]]
                start_end_couples.append([anomalies_frames[0], anomalies_frames[1]])
                start_end_couples.append([anomalies_frames[2], anomalies_frames[3]])
                start_end_couples = torch.from_numpy(np.array(start_end_couples))
                file = items[0].split(".")[0]
                file = file.replace("/", os.sep)
                feature_path = os.path.join(features_path, file)
                length = int(items[1])

                features_list.append((feature_path, start_end_couples, length))

        return features_list
