import logging
import os
import random
from os import path
import numpy as np
import torch
from torch.utils import data
from feature_extractor import read_features


class FeaturesLoader(data.Dataset):
    def __init__(self,
                 features_path,
                 annotation_path,
                 name="<NO_NAME>",
                 shuffle_list_seed=None):

        super(FeaturesLoader, self).__init__()
        self.i = 0
        self.features_path = features_path
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        self.is_shuffled = shuffle_list_seed
        # load video list
        self.state = 'Normal'
        self.features_list_normal, self.features_list_anomaly = FeaturesLoader._get_features_list(
            features_path=self.features_path,
            annotation_path=annotation_path)
        existing_features = set(self.get_existing_features())
        self.features_list_normal = list(existing_features.intersection(self.features_list_normal))
        self.features_list_anomaly = list(existing_features.intersection(self.features_list_anomaly))
        self.normal_i, self.anomalous_i = 0, 0
        if self.is_shuffled is not None:
            self.features_list_anomaly = np.random.permutation(self.features_list_anomaly)
            self.features_list_normal = np.random.permutation(self.features_list_normal)

        logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(
            self.features_list_normal) + len(self.features_list_anomaly)))

    def __len__(self):
        # return len(self.features_list_normal) + len(self.features_list_anomaly)
        return 60

    def __getitem__(self, index):
        succ = False
        while not succ:
            try:
                feature, label = self.get_feature(index)
                succ = True
            except Exception as e:
                index = self.rng.choice(range(0, self.__len__()))
                logging.warning("VideoIter:: ERROR!! (Force using another index:\n{})\n{}".format(index, e))

        return feature, label

    def get_existing_features(self):
        res = []
        for dir in os.listdir(self.features_path):
            dir = path.join(self.features_path, dir)
            if path.isdir(dir):
                for file in os.listdir(dir):
                    file_no_ext = file.split('.')[0]
                    res.append(path.join(dir, file_no_ext))
        return res

    def get_feature(self, index):
        if self.state == 'Normal':  # Load a normal video
            if len(self.features_list_normal) > 0:
                idx = random.randint(0, len(self.features_list_normal)-1)
                feature_subpath = self.features_list_normal[idx].split(os.sep)

            else:  # No normal videos, load an anomalous one
                idx = random.randint(0, len(self.features_list_anomaly)-1)
                feature_subpath = self.features_list_anomaly[idx].split(os.sep)

        elif self.state == 'Anomalous':  # Load an anomalous video
            if len(self.features_list_anomaly) > 0:
                idx = random.randint(0, len(self.features_list_anomaly)-1)
                feature_subpath = self.features_list_anomaly[idx].split(os.sep)

            else:  # No anomalous videos, load a normal one
                idx = random.randint(0, len(self.features_list_normal)-1)
                feature_subpath = self.features_list_normal[idx].split(os.sep)

        features = read_features(dir=os.sep.join(feature_subpath[:-1]),
                                 video_name=feature_subpath[-1])
        
        self.state = 'Anomalous' if self.state == 'Normal' else 'Normal'
        feature_subpath = os.sep.join(feature_subpath)
        label = 1 if "Normal" not in feature_subpath else 0

        return features, label

    @staticmethod
    def _get_features_list(features_path, annotation_path):
        assert os.path.exists(features_path)
        v_id = 0
        features_list_normal = []
        features_list_anomaly = []
        with open(annotation_path, 'r') as f:
            lines = f.read().splitlines(keepends=False)
            for line in lines:
                items = line.split()
                file = items[0].split('.')[0]
                file = file.replace('/', os.sep)
                feature_path = os.path.join(features_path, file)
                if 'Normal' in feature_path:
                    features_list_normal.append(feature_path)
                else:
                    features_list_anomaly.append(feature_path)

        return features_list_normal, features_list_anomaly


class FeaturesLoaderVal(data.Dataset):
    def __init__(self,
                 features_path,
                 annotation_path,
                 name="<NO_NAME>",
                 shuffle_list_seed=None):

        super(FeaturesLoaderVal, self).__init__()
        self.err = 0
        self.features_path = features_path
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        self.is_shuffled = shuffle_list_seed
        # load video list
        self.state = 'Normal'
        self.features_list = FeaturesLoaderVal._get_features_list(
            features_path=self.features_path,
            annotation_path=annotation_path)

        if self.is_shuffled is not None:
            self.features_list_anomaly = np.random.permutation(self.features_list_anomaly)
            self.features_list_normal = np.random.permutation(self.features_list_normal)

        logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(
            self.features_list)))

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, index):
        succ = False
        while not succ:
            try:
                feature, start_end_couples, feature_subpath, length = self.get_feature(index)
                succ = True
            except Exception as e:
                self.err += 1
                print(self.err)
                index = self.rng.choice(range(0, self.__len__()))
                logging.warning("VideoIter:: ERROR!! (Force using another index:\n{})\n{}".format(index, e))

        return feature, start_end_couples, feature_subpath, length

    def get_existing_features(self):
        res = []
        for dir in os.listdir(self.features_path):
            dir = path.join(self.features_path, dir)
            if path.isdir(dir):
                for file in os.listdir(dir):
                    file_no_ext = file.split('.')[0]
                    res.append(path.join(dir, file_no_ext))
        return res

    def get_feature(self, index):
        feature_subpath, start_end_couples, length = self.features_list[index]
        feature_subpath = feature_subpath.split(os.sep)
        features = read_features(dir=os.sep.join(feature_subpath[:-1]),
                                 video_name=feature_subpath[-1])

        feature_subpath = os.sep.join(feature_subpath)

        vid_subpath = feature_subpath.split(os.sep)
        vid_subpath = os.sep.join(vid_subpath[-2:])

        return features, start_end_couples, vid_subpath, length

    @staticmethod
    def _get_features_list(features_path, annotation_path):
        assert os.path.exists(features_path)
        v_id = 0
        features_list = []
        with open(annotation_path, 'r') as f:
            lines = f.read().splitlines(keepends=False)
            for line in lines:
                start_end_couples = []
                items = line.split()
                anomalies_frames = [int(x) for x in items[3:]]
                start_end_couples.append([anomalies_frames[0], anomalies_frames[1]])
                start_end_couples.append([anomalies_frames[2], anomalies_frames[3]])
                start_end_couples = torch.from_numpy(np.array(start_end_couples))
                file = items[0].split('.')[0]
                file = file.replace('/', os.sep)
                feature_path = os.path.join(features_path, file)
                length = int(items[1])

                features_list.append((feature_path, start_end_couples, length))

        return features_list
