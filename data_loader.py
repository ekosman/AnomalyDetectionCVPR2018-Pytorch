""""This module contains a video loader."""

import logging
import os
import sys
from typing import List, Tuple, Union

import numpy as np
from torch import Tensor
from torch.utils import data
from torchvision.datasets.video_utils import VideoClips


class VideoIter(data.Dataset):
    """This class implements a loader for videos."""

    def __init__(
        self,
        clip_length,
        frame_stride,
        dataset_path=None,
        video_transform=None,
        return_label=False,
    ) -> None:
        super().__init__()
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
        """Retrieve the number of the videos in the dataset."""
        return len(self.video_list)

    def getitem_from_raw_video(
        self, idx: int
    ) -> Union[Tuple[Tensor, int, str, str], Tuple[Tensor, int, int, str, str]]:
        """Fetch a sample from the dataset.

        Args:
            idx (int): Index of the sample the retrieve.

        Returns:
            Tuple[Tensor, int, str, str]: Video clip, clip idx in the video, directory name, and file
            Tuple[Tensor, int, int, str, str]: Video clip, label, clip idx in the video, directory name, and file
        """
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
                if trace_back is not None:
                    line = str(trace_back.tb_lineno)
                else:
                    line = "no-line"
                # pylint: disable=line-too-long
                logging.warning(
                    f"VideoIter:: ERROR (line number {line}) !! (Force using another index:\n{index})\n{e}"
                )

        return batch

    def _get_video_list(self, dataset_path: str) -> List[str]:
        """Fetche all videos in a directory and sub-directories.

        Args:
            dataset_path (str): A string that represents the directory of the dataset.

        Raises:
            FileNotFoundError: The directory could not be found in the provided path.

        Returns:
            List[str]
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"VideoIter:: failed to locate: `{dataset_path}'")

        vid_list = []
        for path, _, files in os.walk(dataset_path):
            for name in files:
                if "mp4" not in name:
                    continue
                vid_list.append(os.path.join(path, name))

        logging.info(f"Found {len(vid_list)} video files in {dataset_path}")
        return vid_list


class SingleVideoIter(VideoIter):
    """Loader for a single video."""

    def __init__(
        self,
        clip_length,
        frame_stride,
        video_path,
        video_transform=None,
        return_label=False,
    ) -> None:
        super().__init__(
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
