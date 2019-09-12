import os
import cv2
import numpy as np

import torch.utils.data as data
import logging
import torch
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
                 frame_interval,
                 video_transform=None,
                 name="<NO_NAME>",
                 return_item_subpath=False,
                 shuffle_list_seed=None):
        super(VideoIterTrain, self).__init__()
        # load params
        """
        self.sampler = sampler.SequentialSampling(num=clip_length,
                                               interval=frame_interval,
                                               fix_cursor=True,
                                               shuffle=False)
        """
        seed = 0
        self.sampler = sampler.RandomSampling(num=clip_length,
                                              interval=1,
                                              speed=[1.0, 1.0],
                                              seed=(seed + 0))

        self.force_color = True
        self.dataset_path = dataset_path
        self.video_transform = video_transform
        self.return_item_subpath = return_item_subpath
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        # load video list
        self.video_list = self._get_video_list(dataset_path=self.dataset_path, annotation_path=annotation_path)
        if shuffle_list_seed is not None:
            self.rng.shuffle(self.video_list)
        logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(self.video_list)))

    """    
    def count_frames(self,path):
        
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return frame_count#length
    """

    def getitem_from_raw_video(self, index):
        # get current video info
        # v_id, label, vid_subpath, frame_count = self.video_list[index]
        v_id, vid_subpath, index2startsample = self.video_list[index]
        # del self.video_list[index]
        sampled_idxs = list(range(index2startsample, index2startsample + 16))

        video_path = vid_subpath
        with Video(vid_path=video_path) as video:

            sampled_frames = video.extract_frames(idxs=sampled_idxs, force_color=self.force_color)

            clip_input = np.concatenate(sampled_frames, axis=2)
            if self.video_transform is not None:
                clip_input = self.video_transform(clip_input)

        if "Normal" not in vid_subpath:
            label = 1
        else:
            label = 0

        dir, file = vid_subpath.split(os.sep)[-2:]
        file = file.split('.')[0]
        return clip_input, label, sampled_idxs[0], dir, file

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
                clip = list(range(0, int(items[1]), 16))
                for c in range(len(clip) - 2):
                    vid_list.append((v_id, path.strip('\n'), clip[c]))

                v_id = v_id + 1
        return set(vid_list)


class VideoIterVal(data.Dataset):

    def __init__(self,
                 dataset_path,
                 annotation_path,
                 clip_length,
                 frame_interval,
                 video_transform=None,
                 name="<NO_NAME>",
                 return_item_subpath=False,
                 shuffle_list_seed=None):
        super(VideoIterVal, self).__init__()
        # load params
        self.sampler = sampler.SequentialSampling(num=clip_length,
                                                  interval=1,
                                                  fix_cursor=True,
                                                  shuffle=False)
        self.force_color = True
        self.dataset_path = dataset_path
        self.video_transform = video_transform
        self.return_item_subpath = return_item_subpath
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        # load video list
        self.video_list = self._get_video_list(dataset_path=self.dataset_path, annotation_path=annotation_path)
        if shuffle_list_seed is not None:
            self.rng.shuffle(self.video_list)
        logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(self.video_list)))

    """    
    def count_frames(self,path):
        
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return frame_count#length
    """

    def getitem_from_raw_video(self, index):
        # get current video info
        # v_id, label, vid_subpath, frame_count = self.video_list[index]
        v_id, vid_subpath, index2startsample, start_end_couples = self.video_list[index]
        # del self.video_list[index]
        sampled_idxs = list(range(index2startsample, index2startsample + 16))

        video_path = vid_subpath
        with Video(vid_path=video_path) as video:
            sampled_frames = video.extract_frames(idxs=sampled_idxs, force_color=self.force_color)
            clip_input = np.concatenate(sampled_frames, axis=2)
            if self.video_transform is not None:
                clip_input = self.video_transform(clip_input)

        if "Normal" not in vid_subpath:
            label = 1
        else:
            label = 0

        dir, file = vid_subpath.split(os.sep)[-2:]
        file = file.split('.')[0]
        return clip_input, label, sampled_idxs[0], dir, file

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
                start_end_couples = []
                items = line.split()
                start_end_couples.append((items[3], items[4]))
                if items[5] != -1:
                    start_end_couples.append((items[5], items[6]))
                path = os.path.join(dataset_path, items[0])
                clip = list(range(0, int(items[1]), 16))
                for c in range(len(clip) - 2):
                    vid_list.append((v_id, path.strip('\n'), clip[c], start_end_couples))

                v_id = v_id + 1
        return vid_list
