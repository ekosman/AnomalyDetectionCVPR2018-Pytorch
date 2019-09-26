import os
import cv2
import numpy as np

import torch.utils.data as data
import logging
import torch
from torchvision.datasets.video_utils import VideoClips

import video_sampler as sampler


class Video(object):
    """basic Video class"""

    def __init__(self, vid_path):
        self.open(vid_path)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def reset(self):
        self.close()
        self.vid_path = None
        self.frame_count = -1
        self.faulty_frame = None
        return self

    def open(self, vid_path):
        assert os.path.exists(vid_path)  # , "VideoIter:: cannot locate: `{}'".format(vid_path)

        # close previous video & reset variables
        self.reset()

        # try to open video
        cap = cv2.VideoCapture(vid_path)
        if cap.isOpened():
            self.cap = cap
            self.vid_path = vid_path
        else:
            raise IOError("VideoIter:: failed to open video: `{}'".format(vid_path))

        return self

    def extract_frames(self, idxs, force_color=True):

        frames = self.extract_frames_fast(idxs, force_color)
        if frames is None:
            # try slow method:
            frames = self.extract_frames_slow(idxs, force_color)
        return frames

    def extract_frames_fast(self, idxs, force_color=True):
        assert self.cap is not None, "No opened video."
        if len(idxs) < 1:
            return []

        frames = []
        pre_idx = max(idxs)
        for idx in idxs:
            assert (self.frame_count < 0) or (idx < self.frame_count), \
                "idxs: {} > total valid frames({})".format(idxs, self.frame_count)
            if pre_idx != (idx - 1):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            res, frame = self.cap.read()  # in BGR/GRAY format
            pre_idx = idx
            if not res:
                self.faulty_frame = idx
                return None
            if len(frame.shape) < 3:
                if force_color:
                    # Convert Gray to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return frames

    def extract_frames_slow(self, idxs, force_color=True):
        assert self.cap is not None, "No opened video."
        if len(idxs) < 1:
            return []

        frames = [None] * len(idxs)
        idx = min(idxs)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        while idx <= max(idxs):
            res, frame = self.cap.read()  # in BGR/GRAY format
            if not res:
                # end of the video
                self.faulty_frame = idx
                return None
            if idx in idxs:
                # fond a frame
                if len(frame.shape) < 3:
                    if force_color:
                        # Convert Gray to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pos = [k for k, i in enumerate(idxs) if i == idx]
                for k in pos:
                    frames[k] = frame
            idx += 1
        return frames

    def close(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        return self


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
        # load params

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
