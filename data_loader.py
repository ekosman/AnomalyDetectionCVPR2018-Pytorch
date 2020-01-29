import logging
import os
import numpy as np
import torch.utils.data as data
from torchvision.datasets.video_utils import VideoClips


class VideoIterTrain(data.Dataset):

    def __init__(self,
                 dataset_path,
                 annotation_path,
                 clip_length,
                 frame_stride,
                 video_transform=None,
                 name="<NO_NAME>",
                 return_item_subpath=False,
                 shuffle_list_seed=None):
        super(VideoIterTrain, self).__init__()

        self.force_color = True
        self.dataset_path = dataset_path
        self.frames_stride = frame_stride
        self.video_transform = video_transform
        self.return_item_subpath = return_item_subpath
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        # load video list
        self.video_list = self._get_video_list(dataset_path=self.dataset_path, annotation_path=annotation_path)
        self.total_clip_length_in_frames = clip_length * frame_stride
        self.video_clips = VideoClips(video_paths=self.video_list,
                                      clip_length_in_frames=self.total_clip_length_in_frames,
                                      frames_between_clips=self.total_clip_length_in_frames)
        logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(self.video_list)))

    def getitem_from_raw_video(self, idx):
        # get current video info
        video, _, _, _ = self.video_clips.get_clip(idx)
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video_path = self.video_clips.video_paths[video_idx]
        if self.video_transform is not None:
            video = self.video_transform(video)

        if "Normal" not in video_path:
            label = 1
        else:
            label = 0

        dir, file = video_path.split(os.sep)[-2:]
        file = file.split('.')[0]
        in_clip_frames = list(range(0, self.total_clip_length_in_frames, self.frames_stride))
        return video[in_clip_frames], label, clip_idx, dir, file

    def __getitem__(self, index):
        succ = False
        while not succ:
            try:
                clip_input, label, sampled_idx, dir, file = self.getitem_from_raw_video(index)
                succ = True
            except Exception as e:
                index = self.rng.choice(range(0, self.__len__()))
                logging.warning("VideoIter:: ERROR!! (Force using another index:\n{})\n{}".format(index, e))

        return clip_input, label, sampled_idx, dir, file

    def __len__(self):
        return len(self.video_list)

    def _get_video_list(self, dataset_path, annotation_path):

        assert os.path.exists(dataset_path)  # , "VideoIter:: failed to locate: `{}'".format(dataset_path)
        assert os.path.exists(annotation_path)  # , "VideoIter:: failed to locate: `{}'".format(annotation_path)
        vid_list = []
        with open(annotation_path, 'r') as f:
            for line in f:
                items = line.split()

                path = os.path.join(dataset_path, items[0])
                vid_list.append(path.strip('\n'))
        return set(vid_list)


class VideoIterVal(data.Dataset):

    def __init__(self,
                 dataset_path,
                 annotation_path,
                 clip_length,
                 frame_stride,
                 video_transform=None,
                 name="<NO_NAME>",
                 return_item_subpath=False,
                 shuffle_list_seed=None):
        super(VideoIterVal, self).__init__()
        # load params
        self.frames_stride = frame_stride
        self.dataset_path = dataset_path
        self.video_transform = video_transform
        self.return_item_subpath = return_item_subpath
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        # load video list
        self.video_list = self._get_video_list(dataset_path=self.dataset_path, annotation_path=annotation_path)
        self.total_clip_length_in_frames = clip_length * frame_stride
        self.video_clips = VideoClips(video_paths=self.video_list,
                                      clip_length_in_frames=self.total_clip_length_in_frames,
                                      frames_between_clips=self.total_clip_length_in_frames)
        logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(self.video_list)))

    def getitem_from_raw_video(self, idx):
        # get current video info
        video, _, _, _ = self.video_clips.get_clip(idx)
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video_path = self.video_clips.video_paths[video_idx]
        if self.video_transform is not None:
            video = self.video_transform(video)

        if "Normal" not in video_path:
            label = 1
        else:
            label = 0

        dir, file = video_path.split(os.sep)[-2:]
        file = file.split('.')[0]
        in_clip_frames = list(range(0, self.total_clip_length_in_frames, self.frames_stride))
        return video[in_clip_frames], label, clip_idx, dir, file

    def __getitem__(self, index):
        succ = False
        while not succ:
            try:
                clip_input, label, sampled_idx, dir, file = self.getitem_from_raw_video(index)
                succ = True
            except Exception as e:
                index = self.rng.choice(range(0, self.__len__()))
                logging.warning("VideoIter:: ERROR!! (Force using another index:\n{})\n{}".format(index, e))

        return clip_input, label, sampled_idx, dir, file

    def __len__(self):
        return len(self.video_list)

    def _get_video_list(self, dataset_path, annotation_path):
        assert os.path.exists(dataset_path)  # , "VideoIter:: failed to locate: `{}'".format(dataset_path)
        assert os.path.exists(annotation_path)  # , "VideoIter:: failed to locate: `{}'".format(annotation_path)
        v_id = 0
        vid_list = []
        with open(annotation_path, 'r') as f:
            for line in f:
                items = line.split()
                path = os.path.join(dataset_path, items[0])
                vid_list.append(path.strip('\n'))
        return vid_list
