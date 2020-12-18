import argparse
import logging
import sys
from os import path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_wrapper as pw
import torch
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, \
    QSlider, QStyle, QSizePolicy, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from tqdm import tqdm

from data_loader import SingleVideoIter
from feature_extractor import to_segments
from network.anomaly_detector_model import AnomalyDetector
from network.c3d import C3D
from utils.utils import build_transforms


def get_args():
    parser = argparse.ArgumentParser(description="Video Demo For Anomaly Detection")

    parser.add_argument('--feature_extractor',
                        required=True,
                        help='path to the 3d model for feature extraction')
    parser.add_argument('--feature_method',
                        default='c3d',
                        choices=['c3d', 'mfnet'],
                        help='method to use for feature extraction')
    parser.add_argument('--ad_model',
                        required=True,
                        help="path to the trained AD model")
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
        for data in tqdm(data_iter):
            outputs = model(data.to(device)).detach().cpu()
            features = torch.cat([features, outputs])

    features = features.numpy()
    return to_segments(features, n_segments)


def ad_prediction(model, features, device='cuda'):
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


class Window(QWidget):
    """
    Anomaly detection gui
    Based on media player code from: https://codeloop.org/python-how-to-create-media-player-in-pyqt5/
    """

    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.setWindowTitle("Anomaly Media Player")
        self.setGeometry(350, 100, 700, 500)
        self.setWindowIcon(QIcon('player.png'))

        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.init_ui()

        self.y_pred = None
        self.duration = None

        self.show()

    def init_ui(self):
        # create media player object
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # create videowidget object
        videowidget = QVideoWidget()

        # create open button
        openBtn = QPushButton('Open Video')
        openBtn.clicked.connect(self.open_file)

        # create button for playing
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)

        # create slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)

        # create label
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # create hbox layout
        hboxLayout = QHBoxLayout()
        hboxLayout.setContentsMargins(0, 0, 0, 0)

        # set widgets to the hbox layout
        hboxLayout.addWidget(openBtn)
        hboxLayout.addWidget(self.playBtn)
        hboxLayout.addWidget(self.slider)

        # AD signal
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # create vbox layout
        vboxLayout = QVBoxLayout()
        vboxLayout.addWidget(self.canvas)
        vboxLayout.addWidget(videowidget)
        vboxLayout.addLayout(hboxLayout)
        vboxLayout.addWidget(self.label)

        self.setLayout(vboxLayout)

        self.mediaPlayer.setVideoOutput(videowidget)

        # media player signals
        self.mediaPlayer.stateChanged.connect(self.mediastate_changed)

        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.positionChanged.connect(self.plot)

        self.mediaPlayer.durationChanged.connect(self.duration_changed)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        print(filename)
        if filename != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.playBtn.setEnabled(True)

            features = features_extraction(video_path=filename,
                                           model=feature_extractor,
                                           device=self.device,
                                           n_segments=args.n_segments, )

            y_pred = ad_prediction(model=anomaly_detector,
                                   features=features,
                                   device=self.device, )

            self.y_pred = y_pred.numpy().flatten()

    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediastate_changed(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause)
            )

        else:
            self.playBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay)

            )

    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)
        self.duration = duration

        if self.y_pred is not None:
            self.y_pred = np.repeat(self.y_pred, duration // len(self.y_pred))

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def handle_errors(self):
        self.playBtn.setEnabled(False)
        self.label.setText("Error: " + self.mediaPlayer.errorString())

    def plot(self, position):
        if self.y_pred is not None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_xlim(0, self.mediaPlayer.duration())
            ax.set_ylim(-0.5, 1.5)
            ax.plot(self.y_pred[:position], '*-')
            self.canvas.draw()


if __name__ == '__main__':
    args = get_args()

    anomaly_detector, feature_extractor = load_models(args.feature_extractor,
                                                      args.ad_model,
                                                      features_method=args.feature_method,
                                                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), )

    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
