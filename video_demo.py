import argparse
import logging
from os import path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_wrapper as pw
import torch
from tqdm import tqdm

from data_loader import SingleVideoIter
from feature_extractor import to_segments
from network.anomaly_detector_model import AnomalyDetector
from network.c3d import C3D
from utils.utils import build_transforms, register_logger


def get_args():
	parser = argparse.ArgumentParser(description="Video demo maker")

	parser.add_argument('--video_path',
						required=True,
						help="path of the video to be used for demo")
	parser.add_argument('--feature_extractor',
						required=True,
						help='path to the 3d model for feature extraction')
	parser.add_argument('--feature_method',
						default='c3d',
						choices=['c3d', 'i3d'],
						help='method to use for feature extraction')
	parser.add_argument('--ad_model',
						required=True,
						help="path to the tarined AD model")
	parser.add_argument('--n_segments',
						type=int,
						default=32,
						help='number of segments to use for features averaging')

	return parser.parse_args()


def load_models(feature_extractor_path, ad_model_path, features_method='c3d', device='cuda'):
	"""
	Loads both feature extractor and anomaly detector from the given paths
	:param feature_extractor_path: path of the features extractor weights to load
	:param ad_model_path: path of the anomaly detector weights to load
	:param features_method: name of the model to use for features extraction
	:param device: device to use for the models
	:return: anomaly_detector, feature_extractor
	"""
	assert path.exists(feature_extractor_path)
	assert path.exists(ad_model_path)

	feature_extractor, anomaly_detector = None, None

	if features_method == 'c3d':
		logging.info(f"Loading feature extractor from {feature_extractor_path}")
		feature_extractor = C3D(pretrained=feature_extractor_path)
	else:
		raise NotImplementedError(f"Features extraction method {features_method} not implemented")

	logging.info(f"Loading anomaly detector from {ad_model_path}")
	feature_extractor = feature_extractor.to(device).eval()
	anomaly_detector = pw.System(model=AnomalyDetector(), device=device)
	anomaly_detector.load_model_state(ad_model_path)
	anomaly_detector = anomaly_detector.model.eval()

	return anomaly_detector, feature_extractor


def figure2opencv(figure):
	figure.canvas.draw()
	img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	return img


def features_extraction(video_path, model, device, batch_size=1, frame_stride=1, clip_length=16, n_segments=32):
	"""
	Extracts features of the video. The returned features will be returned after averaging over the required number of
	video segments.
	:param video_path: path of the video to predict
	:param model: model to use for feature extraction
	:param device: device to use for loading data
	:param batch_size: batch size to use for loading data
	:param frame_stride: interval between frames to load
	:param clip_length: number of frames to use for loading each video sample
	:param n_segments: how many chunks the video should be divided into
	:return: features list (n_segments, feature_dim), usually (32, 4096) as in the original paper
	"""
	data_loader = SingleVideoIter(clip_length=clip_length,
							frame_stride=frame_stride,
							video_path=video_path,
							video_transform=build_transforms(),
							return_label=False)
	data_iter = torch.utils.data.DataLoader(data_loader,
											batch_size=batch_size,
											shuffle=False,
											num_workers=0,  # 4, # change this part accordingly
											pin_memory=True)

	logging.info("Extracting features...")
	features = torch.tensor([])
	with torch.no_grad():
		for data, clip_idxs, dirs, vid_names in tqdm(data_iter):
			outputs = model(data.to(device)).detach().cpu()
			features = torch.cat([features, outputs])

	features = features.numpy()
	return to_segments(features, n_segments)


def ad_perdiction(model, features, device='cuda'):
	"""
	Creates frediction for the given feature vectors
	:param model: model to use for anomaly detection
	:param features: features of the video clips
	:param device: device to use for loading the features
	:return: anomaly predictions for the video segments
	"""
	logging.info(f"Performing anomaly detection...")
	features = torch.tensor(features).to(device)
	with torch.no_grad():
		preds = model(features)

	return preds


def gui(video_path, y_pred, save="Just save", s_path="output_video"):
	DISPLAY_IMAGE_SIZE = 500
	BORDER_SIZE = 50  # 100
	FIGHT_BORDER_COLOR = (0, 0, 255)
	NO_FIGHT_BORDER_COLOR = (0, 255, 0)

	plot_range = 100
	# violenceDetector = ViolenceDetector()
	videoReader = cv2.VideoCapture(video_path)
	isCurrentFrameValid, currentImage = videoReader.read()

	if save or save == "Just save":
		# fps = videoReader.get(cv2.CV_CAP_PROP_FPS)
		fps = videoReader.get(cv2.CAP_PROP_FPS)
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(s_path + '.avi', fourcc, fps, (600, 300))

	# farme_window=[0]*clip_length
	fig = plt.figure()
	farme_cout = 0
	length = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))
	while isCurrentFrameValid:
		farme_cout = farme_cout + 1

		targetSize = DISPLAY_IMAGE_SIZE - 2 * BORDER_SIZE
		currentImage = cv2.resize(currentImage, (targetSize, targetSize))

		NO_FIGHT_BORDER_COLOR = NO_FIGHT_BORDER_COLOR
		resultImage = cv2.copyMakeBorder(currentImage,
										 BORDER_SIZE,
										 BORDER_SIZE,
										 BORDER_SIZE,
										 BORDER_SIZE,
										 cv2.BORDER_CONSTANT,
										 value=NO_FIGHT_BORDER_COLOR)

		resultImage = cv2.resize(resultImage, (300, 300))

		# fig = plt.figure()
		# print(pec_store)
		# print(farme_cout)
		# print(y_pred[farme_cout-1])
		plt.plot(farme_cout, y_pred[farme_cout - 1], color='green', marker='o', linestyle='-', linewidth=2,
				 markersize=2)
		# plt.plot(it_uesed, pec_store2, c="r")
		# plt.xlim()
		plt.ylim(0, 1.0)
		if farme_cout < 100:
			plt.xlim(0, 100)
		else:
			plt.xlim(farme_cout - 100, farme_cout + 100)

		plt.xlim(0, length)
		plot_img = figure2opencv(fig)
		plot_img = cv2.resize(plot_img, (300, 300))  # (0, 0), None, .25, .25)

		resultImage = np.concatenate((resultImage, plot_img), axis=1)

		if save == True or save == "Just save":
			# Write the frame into the file 'output.avi'
			print("saving")
			out.write(resultImage)

		if save != "Just save":
			cv2.imshow("Violence Detection", resultImage)
		else:
			print(str(farme_cout) + "/" + str(length))

		userResponse = cv2.waitKey(1)
		if userResponse == ord('q'):
			videoReader.release()
			cv2.destroyAllWindows()
			break

		else:
			isCurrentFrameValid, currentImage = videoReader.read()

	if save or save == "Just save":
		videoReader.release()
		out.release()

	cv2.destroyAllWindows()


# results and video play at the same time
if __name__ == '__main__':
	args = get_args()
	register_logger()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	anomaly_detector, feature_extractor = load_models(args.feature_extractor,
													args.ad_model,
													features_method=args.feature_method,
													device=device,)

	features = features_extraction(video_path=args.video_path,
									model=feature_extractor,
									device=device,
									n_segments=args.n_segments,)

	y_pred = ad_perdiction(model=anomaly_detector,
							features=features,
							device=device,)

	gui(video_path=args.video_path,
		y_pred=y_pred,
		save="Just save",
		s_path=features_dir + dir_list[0][0] + "/" + dir_list[0][1] + "_demoe")
