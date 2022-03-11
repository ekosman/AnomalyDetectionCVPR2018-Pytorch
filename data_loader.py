import logging
import os
import sys
from typing import List, Tuple

import numpy as np
from torch import Tensor
import torch.utils.data as data
from torchvision.datasets.video_utils import VideoClips


class VideoIter(data.Dataset):
    def __init__(
        self,
        clip_length,
        frame_stride,
        dataset_path=None,
        video_transform=None,
        return_label=False,
    ) -> None:
        super(VideoIter, self).__init__()
        # video clip properties
        self.frames_stride = frame_stride
        self.total_clip_length_in_frames = clip_length * frame_stride
        self.video_transform = video_transform

        # IO
        self.dataset_path = dataset_path
        self.video_list = self._get_video_list(dataset_path=self.dataset_path)
        self.return_label = return_label

        # data loading
        self.video_clips = VideoClips(
            video_paths=self.video_list,
            clip_length_in_frames=self.total_clip_length_in_frames,
            frames_between_clips=self.total_clip_length_in_frames,
        )

    @property
    def video_count(self) -> int:
        return len(self.video_list)

    def getitem_from_raw_video(self, idx: int) -> Tuple[Tensor, int, str, str]:
        video, _, _, _ = self.video_clips.get_clip(idx)
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video_path = self.video_clips.video_paths[video_idx]
        in_clip_frames = list(
            range(0, self.total_clip_length_in_frames, self.frames_stride)
        )
        video = video[in_clip_frames]
        if self.video_transform is not None:
            video = self.video_transform(video)

        dir, file = video_path.split(os.sep)[-2:]
        file = file.split(".")[0]

        if self.return_label:
            label = 0 if "Normal" in video_path else 1
            return video, label, clip_idx, dir, file

        return video, clip_idx, dir, file

    def __len__(self) -> int:
        return len(self.video_clips)

    def __getitem__(self, index: int):
        succ = False
        while not succ:
            try:
                batch = self.getitem_from_raw_video(index)
                succ = True
            except Exception as e:
                index = np.random.choice(range(0, self.__len__()))
                trace_back = sys.exc_info()[2]
                line = trace_back.tb_lineno
                logging.warning(
                    f"VideoIter:: ERROR (line number {line}) !! (Force using another index:\n{index})\n{e}"
                )

        return batch

    def _get_video_list(self, dataset_path: str) -> List[str]:
        assert os.path.exists(
            dataset_path
        ), "VideoIter:: failed to locate: `{}'".format(dataset_path)
        vid_list = []
        for path, subdirs, files in os.walk(dataset_path):
            for name in files:
                if "mp4" not in name:
                    continue
                vid_list.append(os.path.join(path, name))

        logging.info(f"Found {len(vid_list)} video files in {dataset_path}")
        return vid_list


class SingleVideoIter(VideoIter):
    def __init__(
        self,
        clip_length,
        frame_stride,
        video_path,
        video_transform=None,
        return_label=False,
    ) -> None:
        super(SingleVideoIter, self).__init__(
            clip_length, frame_stride, video_path, video_transform, return_label
        )

    def _get_video_list(self, dataset_path: str) -> List[str]:
        return [dataset_path]

    def __getitem__(self, idx: int) -> Tensor:
        video, _, _, _ = self.video_clips.get_clip(idx)
        in_clip_frames = list(
            range(0, self.total_clip_length_in_frames, self.frames_stride)
        )
        video = video[in_clip_frames]
        if self.video_transform is not None:
            video = self.video_transform(video)

        return video
