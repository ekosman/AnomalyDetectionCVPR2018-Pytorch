import logging
import os
import numpy as np
import torch.utils.data as data
from torchvision.datasets.video_utils import VideoClips
from tqdm import tqdm


class VideoIter(data.Dataset):
    def __init__(self,
                 clip_length,
                 frame_stride,
                 dataset_path=None,
                 annotation_path=None,
                 video_transform=None,
                 name="<NO_NAME>",
                 shuffle_list_seed=None,
                 single_load=False):
        super(VideoIter, self).__init__()
        self.dataset_path = dataset_path
        self.frames_stride = frame_stride
        self.video_transform = video_transform
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)

        # load video list
        if dataset_path is not None:
            self.video_list = self._get_video_list(dataset_path=self.dataset_path)

        elif type(annotation_path)==list():
            self.video_list = annotation_path
        else:
            self.video_list=[annotation_path]

        self.total_clip_length_in_frames = clip_length * frame_stride

        if single_load:
            print("loading each file at a time")
            self.video_clips = VideoClips(video_paths=[self.video_list[0]],
                                          clip_length_in_frames=self.total_clip_length_in_frames,
                                          frames_between_clips=self.total_clip_length_in_frames)
            with tqdm(total=len(self.video_list[1:])+1,desc=' total % of videos loaded') as pbar1:
                for video_list_used in self.video_list[1:]:
                    print(video_list_used)
                    pbar1.update(1)
                    video_clips_out = VideoClips(video_paths=[video_list_used],
                                                 clip_length_in_frames=self.total_clip_length_in_frames,
                                                 frames_between_clips=self.total_clip_length_in_frames)
                    self.video_clips.clips.append(video_clips_out.clips[0])
                    self.video_clips.cumulative_sizes.append(self.video_clips.cumulative_sizes[-1]+video_clips_out.cumulative_sizes[0])
                    self.video_clips.resampling_idxs.append(video_clips_out.resampling_idxs[0])
                    self.video_clips.video_fps.append(video_clips_out.video_fps[0])
                    self.video_clips.video_paths.append(video_clips_out.video_paths[0])
                    self.video_clips.video_pts.append(video_clips_out.video_pts[0])
        else:
            print("single loader used")
            self.video_clips = VideoClips(video_paths=self.video_list,
                                          clip_length_in_frames=self.total_clip_length_in_frames,
                                          frames_between_clips=self.total_clip_length_in_frames)

        logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(self.video_list)))

    def getitem_from_raw_video(self, idx):
        # get current video info
        video, _, _, _ = self.video_clips.get_clip(idx)
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video_path = self.video_clips.video_paths[video_idx]
        in_clip_frames = list(range(0, self.total_clip_length_in_frames, self.frames_stride))
        video = video[in_clip_frames]
        if self.video_transform is not None:
            video = self.video_transform(video)

        label = 0 if "Normal" in video_path else 1

        dir, file = video_path.split(os.sep)[-2:]
        file = file.split('.')[0]

        return video, label, clip_idx, dir, file

    def __len__(self):
        return len(self.video_clips)

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

    @staticmethod
    def _get_video_list(dataset_path):
        assert os.path.exists(dataset_path)  , "VideoIter:: failed to locate: `{}'".format(dataset_path)
        vid_list = []
        for path, subdirs, files in os.walk(dataset_path):
            for name in files:
                vid_list.append(os.path.join(path, name))


        return vid_list
