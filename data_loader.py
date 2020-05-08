import logging
import os
import numpy as np
import torch.utils.data as data
from torchvision.datasets.video_utils import VideoClips

from tqdm import tqdm


class VideoIterTrain(data.Dataset):

    def __init__(self,
                 dataset_path,
                 annotation_path,
                 clip_length,
                 frame_stride,
                 video_transform=None,
                 name="<NO_NAME>",
                 return_item_subpath=False,
                 shuffle_list_seed=None,single_load=False):
        super(VideoIterTrain, self).__init__()

        self.force_color = True
        if dataset_path!=None:
            self.dataset_path = dataset_path
        self.frames_stride = frame_stride
        self.video_transform = video_transform
        self.return_item_subpath = return_item_subpath
        self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
        # load video list
        if dataset_path!=None:
            self.video_list = self._get_video_list(dataset_path=self.dataset_path, annotation_path=annotation_path)

        elif type(annotation_path)==list():
            self.video_list = annotation_path
        else:
            self.video_list=[annotation_path]

        self.total_clip_length_in_frames = clip_length * frame_stride

        #size_list=[]
        if single_load==True:
            print("loading each file at a time")
            self.video_clips = VideoClips(video_paths=[self.video_list[0]],
                                          clip_length_in_frames=self.total_clip_length_in_frames,
                                          frames_between_clips=self.total_clip_length_in_frames)
            with tqdm(total=len(self.video_list[1:])+1,desc=' total % of videos loaded') as pbar1:
                for video_list_used in self.video_list[1:]: #length of load?)
                    #blockPrint()
                    print(video_list_used)
                    import os
                    #print("size "+str(os.path.getsize(video_list_used)))
                    #size_list.append(os.path.getsize(video_list_used))
                    #print(max(size_list))
                    pbar1.update(1)
                    video_clips_out = VideoClips(video_paths=[video_list_used],
                                              clip_length_in_frames=self.total_clip_length_in_frames,
                                              frames_between_clips=self.total_clip_length_in_frames)
                    # if video_list_used =="/media/peter/Maxtor/AD-pytorch/UCF_Crimes/Videos/Training_Normal_Videos_Anomaly/Normal_Videos547_x264.mp4":
                    #     continue
                    # #enablePrint()
                    self.video_clips.clips.append(video_clips_out.clips[0])
                    #print(self.video_clips.cumulative_sizes)
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
        in_clip_frames = list(range(0, self.total_clip_length_in_frames, self.frames_stride))
        video_path = self.video_clips.video_paths[idx]#video_idx
        print(idx)
        print(video_idx)
        print(video_path)
        in_clip_frames = list(range(0, self.total_clip_length_in_frames, self.frames_stride))
        video = video[in_clip_frames]
        if self.video_transform is not None:
            video = self.video_transform(video)

        if "Normal" not in video_path:
            label = 1
        else:
            label = 0

        dir, file = video_path.split(os.sep)[-2:]
        file = file.split('.')[0]

        #video=video.numpy()
        #test=video.shape
        #t=video[:][0]
        #video[in_clip_frames]
        return video, label, clip_idx, dir, file#video[:, in_clip_frames, :, :], label, clip_idx, dir, file

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
        return vid_list #set(vid_list)


# class VideoIterVal(data.Dataset):
#
#     def __init__(self,
#                  dataset_path,
#                  annotation_path,
#                  clip_length,
#                  frame_stride,
#                  video_transform=None,
#                  name="<NO_NAME>",
#                  return_item_subpath=False,
#                  shuffle_list_seed=None,single_load=False):
#         super(VideoIterVal, self).__init__()
#         # load params
#         self.frames_stride = frame_stride
#         self.dataset_path = dataset_path
#         self.video_transform = video_transform
#         self.return_item_subpath = return_item_subpath
#         self.rng = np.random.RandomState(shuffle_list_seed if shuffle_list_seed else 0)
#         # load video list
#         self.video_list = self._get_video_list(dataset_path=self.dataset_path, annotation_path=annotation_path)
#         self.total_clip_length_in_frames = clip_length * frame_stride
#         self.video_clips = VideoClips(video_paths=self.video_list,
#                                       clip_length_in_frames=self.total_clip_length_in_frames,
#                                       frames_between_clips=self.total_clip_length_in_frames)
#         # #size_list=[]
#         # if single_load==True:
#         #     self.video_clips = VideoClips(video_paths=[self.video_list[0]],
#         #                                   clip_length_in_frames=self.total_clip_length_in_frames,
#         #                                   frames_between_clips=self.total_clip_length_in_frames)
#         #     with tqdm(total=len(self.video_list[1:])+1,desc=' total % of videos loaded') as pbar1:
#         #         for video_list_used in self.video_list[1:]: #length of load?)
#         #             #blockPrint()
#         #             #print(video_list_used)
#         #             import os
#         #             #print("size "+str(os.path.getsize(video_list_used)))
#         #             #size_list.append(os.path.getsize(video_list_used))
#         #             #print(max(size_list))
#         #             pbar1.update(1)
#         #             video_clips_out = VideoClips(video_paths=[video_list_used],
#         #                                       clip_length_in_frames=self.total_clip_length_in_frames,
#         #                                       frames_between_clips=self.total_clip_length_in_frames)
#         #             # if video_list_used =="/media/peter/Maxtor/AD-pytorch/UCF_Crimes/Videos/Training_Normal_Videos_Anomaly/Normal_Videos547_x264.mp4":
#         #             #     continue
#         #             # #enablePrint()
#         #             self.video_clips.clips.append(video_clips_out.clips[0])
#         #             self.video_clips.cumulative_sizes.append(video_clips_out.cumulative_sizes[0])
#         #             self.video_clips.resampling_idxs.append(video_clips_out.resampling_idxs[0])
#         #             self.video_clips.video_fps.append(video_clips_out.video_fps[0])
#         #             self.video_clips.video_paths.append(video_clips_out.video_paths[0])
#         #             self.video_clips.video_pts.append(video_clips_out.video_pts[0])
#         # else:
#         #     print("single loader used")
#         #     self.video_clips = VideoClips(video_paths=self.video_list,
#         #                                  clip_length_in_frames=self.total_clip_length_in_frames,
#         #                                  frames_between_clips=self.total_clip_length_in_frames)
#         logging.info("VideoIter:: iterator initialized (phase: '{:s}', num: {:d})".format(name, len(self.video_list)))
#
#     def getitem_from_raw_video(self, idx):
#         # get current video info
#         video, _, _, _ = self.video_clips.get_clip(idx)
#         video_idx, clip_idx = self.video_clips.get_clip_location(idx)
#         video_path = self.video_clips.video_paths[video_idx]
#         in_clip_frames = list(range(0, self.total_clip_length_in_frames, self.frames_stride))
#         video = video[in_clip_frames]
#         if self.video_transform is not None:
#             video = self.video_transform(video)
#
#         if "Normal" not in video_path:
#             label = 1
#         else:
#             label = 0
#
#         dir, file = video_path.split(os.sep)[-2:]
#         file = file.split('.')[0]
#         #in_clip_frames = list(range(0, self.total_clip_length_in_frames, self.frames_stride))
#         return video, label, clip_idx, dir, file #video[in_clip_frames], label, clip_idx, dir, file
#
#     def __getitem__(self, index):
#         succ = False
#         while not succ:
#             try:
#                 clip_input, label, sampled_idx, dir, file = self.getitem_from_raw_video(index)
#                 succ = True
#             except Exception as e:
#                 index = self.rng.choice(range(0, self.__len__()))
#                 logging.warning("VideoIter:: ERROR!! (Force using another index:\n{})\n{}".format(index, e))
#
#         return clip_input, label, sampled_idx, dir, file
#
#     def __len__(self):
#         return len(self.video_list)
#
#     def _get_video_list(self, dataset_path, annotation_path):
#         assert os.path.exists(dataset_path)  # , "VideoIter:: failed to locate: `{}'".format(dataset_path)
#         assert os.path.exists(annotation_path)  # , "VideoIter:: failed to locate: `{}'".format(annotation_path)
#         v_id = 0
#         vid_list = []
#         with open(annotation_path, 'r') as f:
#             for line in f:
#                 items = line.split()
#                 path = os.path.join(dataset_path, items[0])
#                 vid_list.append(path.strip('\n'))
#         return vid_list
