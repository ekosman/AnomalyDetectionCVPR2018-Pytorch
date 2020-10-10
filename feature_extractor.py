import argparse
import logging
import os
from os import path, mkdir
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from data_loader import VideoIter
from network.c3d import C3D
from utils.utils import set_logger, build_transforms


def get_args():
	parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
	# debug
	parser.add_argument('--debug-mode', action='store_true', default=False,
						help="print all setting for debugging.")
	# io
	parser.add_argument('--dataset_path', default='../kinetics2/kinetics2/AnomalyDetection',
						help="path to dataset")
	parser.add_argument('--clip-length', type=int, default=16,
						help="define the length of each input sample.")
	parser.add_argument('--num_workers', type=int, default=32,
						help="define the number of workers used for loading the videos")
	parser.add_argument('--frame-interval', type=int, default=1,
						help="define the sampling interval between frames.")
	parser.add_argument('--log-file', type=str, default="",
						help="set logging file.")
	parser.add_argument('--save_dir', type=str, default="features",
						help="set logging file.")

	# device
	parser.add_argument('--pretrained_3d', type=str,
						default='',
						help="load default 3D pretrained model.")
	parser.add_argument('--resume-epoch', type=int, default=-1,
						help="resume train")
	# optimization
	parser.add_argument('--batch-size', type=int, default=8,
						help="batch size")
	parser.add_argument('--random-seed', type=int, default=1,
						help='random seed (default: 1)')
	return parser.parse_args()


def to_32_segments(data):
	"""
	These code is taken from:
	https://github.com/rajanjitenpatel/C3D_feature_extraction/blob/b5894fa06d43aa62b3b64e85b07feb0853e7011a/extract_C3D_feature.py#L805
	:param data: list of features of a certain video
	:return: list of 32 segments
	"""
	data = np.array(data)
	Segments_Features = []
	thirty2_shots = np.round(np.linspace(0, len(data), num=33))
	for i in range(0, len(thirty2_shots) - 1):
		ss = int(thirty2_shots[i])
		ee = int(thirty2_shots[i + 1]) - 1

		if i == len(thirty2_shots):
			ee = thirty2_shots[i + 1]

		if ss == ee:
			temp_vect = data[ss, :]
		elif ee < ss:
			temp_vect = data[ss, :]
		else:
			temp_vect = data[ss:ee, :].mean(axis=0)

		temp_vect = temp_vect / np.linalg.norm(temp_vect)
		if np.linalg.norm == 0:
			logging.error("Feature norm is 0")
			exit()
		if len(temp_vect) != 0:
			Segments_Features.append(temp_vect.tolist())

	return Segments_Features


current_path = None
current_dir = None
current_data = None


class FeaturesWriter:
	def __init__(self, num_videos, chunk_size=16):
		self.path = None
		self.dir = None
		self.data = None
		self.chunk_size = chunk_size
		self.num_videos = num_videos
		self.dump_count = 0

	def _init_video(self, video_name, dir):
		self.path = path.join(dir, f"{video_name}.txt")
		self.dir = dir
		self.data = dict()

	def has_video(self):
		return self.data is not None

	def dump(self):
		logging.info(f'{self.dump_count} / {self.num_videos}:	Dumping {self.path}')
		self.dump_count += 1
		if not path.exists(self.dir):
			os.mkdir(self.dir)

		features = to_32_segments([self.data[key] for key in sorted(self.data)])
		with open(self.path, 'w') as fp:
			for d in features:
				d = [str(x) for x in d]
				fp.write(' '.join(d) + '\n')

	def _is_new_video(self, video_name, dir):
		new_path = path.join(dir, f"{video_name}.txt")
		if self.path != new_path and self.path is not None:
			return True

		return False

	def store(self, feature, idx):
		self.data[idx] = list(feature)

	def write(self, feature, video_name, idx, dir):
		if not self.has_video():
			self._init_video(video_name, dir)

		if self._is_new_video(video_name, dir):
			self.dump()
			self._init_video(video_name, dir)

		self.store(feature, idx)


def read_features(video_name, dir):
	file_path = f"{video_name}.txt"
	file_path = path.join(dir, file_path)
	if not path.exists(file_path):
		raise Exception(f"Feature doesn't exist: {file_path}")
	features = None
	with open(file_path, 'r') as fp:
		data = fp.read().splitlines(keepends=False)
		features = np.zeros((len(data), 4096))
		for i, line in enumerate(data):
			features[i, :] = [float(x) for x in line.split(' ')]

	return torch.from_numpy(features)


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	args = get_args()
	set_logger(log_file=args.log_file, debug_mode=args.debug_mode)

	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed(args.random_seed)
	cudnn.benchmark = True

	data_loader = VideoIter(dataset_path=args.dataset_path,
							clip_length=args.clip_length,
							frame_stride=args.frame_interval,
							video_transform=build_transforms(),
							return_label=False)

	data_iter = torch.utils.data.DataLoader(data_loader,
											batch_size=args.batch_size,
											shuffle=False,
											num_workers=args.num_workers,
											pin_memory=True)

	network = C3D(pretrained=args.pretrained_3d)
	network = network.to(device)

	if not path.exists(args.save_dir):
		mkdir(args.save_dir)

	features_writer = FeaturesWriter(num_videos=data_loader.video_count)
	with torch.no_grad():
		for data, clip_idxs, dirs, vid_names in data_iter:
			outputs = network(data.to(device)).detach().cpu().numpy()

			for i, (dir, vid_name, clip_idx) in enumerate(zip(dirs, vid_names, clip_idxs)):
				dir = path.join(args.save_dir, dir)
				features_writer.write(feature=outputs[i],
									  video_name=vid_name,
									  idx=clip_idx,
									  dir=dir, )

	features_writer.dump()


if __name__ == "__main__":
	main()
