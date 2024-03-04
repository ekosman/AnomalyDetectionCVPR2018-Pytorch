import argparse
import logging
import sys
from os import path
from typing import List

import numpy as np
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QUrl  # pylint: disable=no-name-in-module
from PyQt5.QtGui import QIcon, QPalette  # pylint: disable=no-name-in-module
from PyQt5.QtMultimedia import (  # pylint: disable=no-name-in-module
    QMediaContent,
    QMediaPlayer,
)
from PyQt5.QtMultimediaWidgets import QVideoWidget  # pylint: disable=no-name-in-module
from PyQt5.QtWidgets import QApplication  # pylint: disable=no-name-in-module
from PyQt5.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyle,
    QWidget,
)
from torch import Tensor
from tqdm import tqdm

from data_loader import SingleVideoIter
from feature_extractor import read_features, to_segments
from network.anomaly_detector_model import AnomalyDetector
from network.TorchUtils import get_torch_device
from utils.load_model import load_models
from utils.types import Device, FeatureExtractor
from utils.utils import build_transforms

APP_NAME = "Anomaly Media Player"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video Demo For Anomaly Detection")

    parser.add_argument(
        "--feature_extractor",
        required=True,
        help="path to the 3d model for feature extraction",
    )
    parser.add_argument(
        "--feature_method",
        default="c3d",
        choices=["c3d", "mfnet", "r3d101", "r3d101"],
        help="method to use for feature extraction",
    )
    parser.add_argument(
        "--ad_model", required=True, help="path to the trained AD model"
    )
    parser.add_argument(
        "--n_segments",
        type=int,
        default=32,
        help="number of segments to use for features averaging",
    )

    return parser.parse_args()


def features_extraction(
    video_path: str,
    model: FeatureExtractor,
    device: Device,
    batch_size: int = 1,
    frame_stride: int = 1,
    clip_length: int = 16,
    n_segments: int = 32,
    progress_bar=None,
) -> List[np.ndarray]:
    """Extracts features of the video.The returned features will be returned
    after averaging over the required number of video segments.

    :param video_path: path of the video to predict
    :param model: model to use for feature extraction
    :param device: device to use for loading data
    :param batch_size: batch size to use for loading data
    :param frame_stride: interval between frames to load
    :param clip_length: number of frames to use for loading each video
        sample
    :param n_segments: how many chunks the video should be divided into
    :param progress_bar: TODO
    :return: features list (n_segments, feature_dim), usually (32, 4096)
        as in the original paper
    """
    data_loader = SingleVideoIter(
        clip_length=clip_length,
        frame_stride=frame_stride,
        video_path=video_path,
        video_transform=build_transforms(mode=args.feature_method),
        return_label=False,
    )
    data_iter = torch.utils.data.DataLoader(
        data_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 4, # change this part accordingly
        pin_memory=True,
    )

    logging.info("Extracting features...")
    features = torch.tensor([])  # pylint: disable=not-callable

    if progress_bar is not None:
        progress_bar.setRange(0, len(data_iter))

    with torch.no_grad():
        for i, data in tqdm(enumerate(data_iter)):
            outputs = model(data.to(device)).detach().cpu()
            features = torch.cat([features, outputs])

            if progress_bar is not None:
                progress_bar.setValue(i + 1)

    features = features.numpy()
    return to_segments(features, n_segments)


def ad_prediction(model: AnomalyDetector, features: Tensor, device: Device) -> Tensor:
    """Creates prediction for the given feature vectors.

    :param model: model to use for anomaly detection
    :param features: features of the video clips
    :param device: device to use for loading the features
    :return: anomaly predictions for the video segments
    """
    logging.info("Performing anomaly detection...")
    features = torch.tensor(features).to(device)  # pylint: disable=not-callable
    with torch.no_grad():
        preds = model(features)

    return preds.detach().cpu().numpy().flatten()


class MplCanvas(FigureCanvasQTAgg):
    # pylint: disable=unused-argument
    def __init__(self, parent=None, width=5, height=4, dpi=100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class Window(QWidget):
    """Anomaly detection gui Based on media player code from:

    https://codeloop.org/python-how-to-create-media-player-in-pyqt5/
    """

    def __init__(self) -> None:
        super().__init__()

        self.device = get_torch_device()

        self.setWindowTitle(APP_NAME)
        self.setGeometry(350, 100, 700, 500)
        self.setWindowIcon(QIcon("player.png"))

        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.init_ui()

        self._y_pred = torch.tensor([])
        self.duration = None

        self.show()

    def init_ui(self):
        # create media player object
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # create videowidget object
        videowidget = QVideoWidget()

        # create open button
        openBtn = QPushButton("Open Video")
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

        # create grid layout
        gridLayout = QGridLayout()

        # AD signal

        self.graphWidget = MplCanvas(self, width=5, height=1, dpi=100)

        # Feature extraction progress bar
        self.pbar = QProgressBar()
        self.pbar.setTextVisible(True)

        # set widgets to the hbox layout
        gridLayout.addWidget(self.graphWidget, 0, 0, 1, 5)
        gridLayout.addWidget(videowidget, 1, 0, 5, 5)
        gridLayout.addWidget(openBtn, 6, 0, 1, 1)
        gridLayout.addWidget(self.playBtn, 6, 1, 1, 1)
        gridLayout.addWidget(self.slider, 6, 2, 1, 3)
        gridLayout.addWidget(self.pbar, 7, 0, 1, 5)
        gridLayout.addWidget(self.label, 7, 2, 1, 1)

        self.setLayout(gridLayout)

        self.mediaPlayer.setVideoOutput(videowidget)

        # media player signals
        self.mediaPlayer.stateChanged.connect(self.mediastate_changed)

        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.positionChanged.connect(self.plot)

        self.mediaPlayer.durationChanged.connect(self.duration_changed)

    def open_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")

        if filename == "":
            return

        feature_load_message_box = QMessageBox()
        feature_load_message_box.setIcon(QMessageBox.Question)
        feature_load_message_box.setText(
            "Extract features from the chosen video file or load from file?"
        )
        feature_load_message_box.setWindowTitle(APP_NAME)
        feature_load_message_box.addButton(
            "Extract features", feature_load_message_box.ActionRole
        )
        feature_load_message_box.addButton(
            "Load features from file", feature_load_message_box.ActionRole
        )
        feature_load_message_box.buttonClicked.connect(self._features_msgbtn)
        feature_load_message_box.exec_()

        if not path.exists(filename):
            raise FileNotFoundError("The chosen file does not exist.")

        if self.feature_source == "Extract features":
            self.label.setText("Extracting features...")

            features = torch.tensor(
                features_extraction(
                    video_path=filename,
                    model=feature_extractor,
                    device=self.device,
                    n_segments=args.n_segments,
                    progress_bar=self.pbar,
                )
            )
            features = torch.tensor(features)

        elif self.feature_source == "Load features from file":
            f_filename, _ = QFileDialog.getOpenFileName(self, "Open Features File")
            features = read_features(file_path=f_filename)

        self._y_pred = ad_prediction(
            model=anomaly_detector,
            features=features,
            device=self.device,
        )

        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
        self.playBtn.setEnabled(True)
        self.label.setText("Done! Click the Play button")

    def _features_msgbtn(self, i) -> None:
        self.feature_source = i.text()  # pylint: disable=attribute-defined-outside-init

    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    # pylint: disable=unused-argument
    def mediastate_changed(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

        else:
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)
        self.duration = duration

        if self._y_pred is not None:
            self._y_pred = np.repeat(self._y_pred, duration // len(self._y_pred))

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def handle_errors(self):
        self.playBtn.setEnabled(False)
        self.label.setText("Error: " + self.mediaPlayer.errorString())

    def plot(self, position):
        if self._y_pred is not None:
            ax = self.graphWidget.axes
            ax.clear()
            ax.set_xlim(0, self.mediaPlayer.duration())
            ax.set_ylim(-0.1, 1.1)
            ax.plot(self._y_pred[:position], "*-", linewidth=7)
            self.graphWidget.draw()


if __name__ == "__main__":
    args = get_args()

    anomaly_detector, feature_extractor = load_models(
        args.feature_extractor,
        args.ad_model,
        features_method=args.feature_method,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
