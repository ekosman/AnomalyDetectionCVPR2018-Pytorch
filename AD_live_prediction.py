""""This module contains a procedure for real time anomaly detection."""

import argparse
import logging
import sys
from typing import List
import cv2

import numpy as np
from numpy.lib.function_base import copy
import torch
from torch import Tensor, nn
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, Qt, pyqtSignal  # pylint: disable=no-name-in-module
from PyQt5.QtGui import QIcon, QPalette, QPixmap  # pylint: disable=no-name-in-module
from PyQt5.QtMultimedia import (  # pylint: disable=no-name-in-module
    QMediaPlayer,
    QCameraInfo,
)  # pylint: disable=no-name-in-module
from PyQt5.QtWidgets import (  # pylint: disable=no-name-in-module
    QApplication,
    QLabel,
    QWidget,
    QPushButton,
    QStyle,
    QGridLayout,
    QComboBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from feature_extractor import to_segments
from network.TorchUtils import get_torch_device
from utils.load_model import load_models
from utils.types import Device
from utils.utils import build_transforms


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
        "--clip_length",
        type=int,
        default=16,
        help="define the length of each input sample",
    )

    return parser.parse_args()


def features_extraction(
    frames, model, device, frame_stride=1, transforms=None
) -> List[np.array]:
    """
    Extracts features of the video. The returned features will be returned after
    averaging over the required number of video segments.

    Args:
        frames: a sequence of video frames to predict
        model: model to use for feature extraction
        device: device to use for loading data
        frame_stride: interval between frames to load

    Returns:
        feature (1, feature_dim), usually (1, 4096) as in the original paper
    """

    frames = torch.tensor(frames)  # pylint: disable=not-callable
    frames = transforms(frames).to(device)
    data = frames[:, range(0, frames.shape[1], frame_stride), ...]
    data = data.unsqueeze(0)

    with torch.no_grad():
        outputs = model(data.to(device)).detach().cpu()

    return to_segments(outputs.numpy(), 1)


def ad_prediction(
    model: nn.Module, features: Tensor, device: Device = "cuda"
) -> np.ndarray:
    """
    Creates frediction for the given feature vectors

    Args:
        model: model to use for anomaly detection
        features: features of the video clips
        device: device to use for loading the features

    Returns:
        Anomaly predictions for the video segments
    """

    logging.info("Performing anomaly detection...")
    features = torch.tensor(features).to(device)  # pylint: disable=not-callable
    with torch.no_grad():
        preds = model(features)

    return preds.detach().cpu().numpy().flatten()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self) -> None:
        super().__init__()
        self._run_flag = True

    def run(self) -> None:
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self) -> None:
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class MplCanvas(FigureCanvasQTAgg):
    # pylint: disable=unused-argument
    def __init__(self, parent=None, width=5, height=4, dpi=100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class Window(QWidget):
    """
    Anomaly detection live gui
    Based on media player code from:
    https://codeloop.org/python-how-to-create-media-player-in-pyqt5/
    """

    def __init__(self, clip_length=16, transforms=None) -> None:
        super().__init__()

        self.camera = None
        self.current_camera_name = None
        self.frames_queue = []

        self.clip_length = clip_length
        self.transforms = transforms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.setWindowTitle("Anomaly Media Player")
        self.setGeometry(350, 100, 700, 500)
        self.setWindowIcon(QIcon("player.png"))

        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.init_ui()

        self.y_pred = [1] * 100

        self.show()

    def init_ui(self) -> None:
        """Create media player object"""

        # setup camera
        self.available_cameras = QCameraInfo.availableCameras()
        self.camera_view = QLabel()
        self.frames_queue = []
        self.camera_id = None
        self.select_camera()

        # creating a combo box for selecting camera
        camera_selector = QComboBox()
        camera_selector.setStatusTip("Choose camera")
        camera_selector.setToolTip("Select Camera")
        camera_selector.setToolTipDuration(2500)
        camera_selector.addItems(
            [camera.description() for camera in self.available_cameras]
        )
        camera_selector.currentIndexChanged.connect(self.select_camera)

        # create button for playing
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(True)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)

        # create grid layout
        gridLayout = QGridLayout()

        # AD signal
        self.graphWidget = MplCanvas(self, width=5, height=1, dpi=100)

        # set widgets to the hbox layout
        gridLayout.addWidget(self.graphWidget, 0, 0, 1, 5)
        gridLayout.addWidget(self.camera_view, 1, 0, 5, 5)

        self.setLayout(gridLayout)

        # create the video capture thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def update_image(self, cv_img) -> None:
        qt_img = self.convert_cv_qt(cv_img)
        self.camera_view.setPixmap(qt_img)
        self.frames_queue.append(cv_img)
        if len(self.frames_queue) == self.clip_length:
            batch = copy(self.frames_queue)
            features = features_extraction(
                frames=batch,
                model=feature_extractor,
                device=self.device,
                transforms=self.transforms,
            )

            new_pred = ad_prediction(
                model=anomaly_detector,
                features=features,
                device=self.device,
            )[0]
            self.y_pred.append(new_pred)
            del self.y_pred[0]
            self.plot()
            self.frames_queue = []

    def convert_cv_qt(self, cv_img: np.ndarray) -> QPixmap:
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        display_height, display_width = (
            self.camera_view.height(),
            self.camera_view.width(),
        )
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(
            display_width, display_height, Qt.KeepAspectRatio
        )
        return QPixmap.fromImage(p)

    def select_camera(self, camera=0) -> None:
        # getting the selected camera
        self.camera = cv2.VideoCapture(camera)

        # getting current camera name
        self.current_camera_name = self.available_cameras[camera].description()

    def play_video(self) -> None:
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediastate_changed(self, *_args) -> None:
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

        else:
            self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def handle_errors(self) -> None:
        self.playBtn.setEnabled(False)
        self.label.setText("Error: " + self.mediaPlayer.errorString())

    def plot(self) -> None:
        ax = self.graphWidget.axes
        ax.clear()
        # ax.set_xlim(0, self.mediaPlayer.duration())
        ax.set_ylim(-0.1, 1.1)
        ax.plot(self.y_pred, "*-", linewidth=7)
        self.graphWidget.draw()


if __name__ == "__main__":
    args = get_args()

    anomaly_detector, feature_extractor = load_models(
        args.feature_extractor,
        args.ad_model,
        features_method=args.feature_method,
        device=get_torch_device(),
    )

    transforms = build_transforms(mode=args.feature_method)

    app = QApplication(sys.argv)
    window = Window(args.clip_length, transforms)

    sys.exit(app.exec_())
