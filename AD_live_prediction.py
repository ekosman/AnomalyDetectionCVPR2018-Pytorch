"""This module contains a procedure for real time anomaly detection."""

import argparse
import logging
import sys
from typing import Callable, List, Optional

import cv2
import numpy as np
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from numpy.lib.function_base import copy
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QThread  # pylint: disable=no-name-in-module
from PyQt5.QtGui import QIcon, QPalette, QPixmap  # pylint: disable=no-name-in-module
from PyQt5.QtMultimedia import (  # pylint: disable=no-name-in-module
    QCameraInfo,
    QMediaPlayer,
)
from PyQt5.QtWidgets import QComboBox  # pylint: disable=no-name-in-module
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QPushButton,
    QStyle,
    QWidget,
)
from torch import Tensor, nn

from feature_extractor import to_segments
from network.TorchUtils import get_torch_device
from utils.load_model import load_models
from utils.stack import Stack
from utils.types import Device, FeatureExtractor
from utils.utils import build_transforms

MAX_PREDS = 50


# pylint disable=missing-function-docstring
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
        choices=["c3d", "mfnet", "r3d101", "r3d152"],
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
    frames,
    model: FeatureExtractor,
    device: Device,
    frame_stride: int = 1,
    transforms: Optional[Callable] = None,
) -> List[np.ndarray]:
    """Extracts features of the video. The returned features will be returned
    after averaging over the required number of video segments.

    Args:
        frames: a sequence of video frames to predict
        model: model to use for feature extraction
        device: device to use for loading data
        frame_stride: interval between frames to load

    Returns:
        feature (1, feature_dim), usually (1, 4096) as in the original paper
    """

    frames = torch.tensor(frames, device=device)  # pylint: disable=not-callable
    if transforms is not None:
        frames = transforms(frames)
    data = frames[:, range(0, frames.shape[1], frame_stride), ...]
    data = data.unsqueeze(0)

    with torch.no_grad():
        outputs = model(data.to(device)).detach().cpu()

    return to_segments(outputs.numpy(), 1)


def ad_prediction(
    model: nn.Module, features: Tensor, device: Device = "cuda"
) -> np.ndarray:
    """Creates frediction for the given feature vectors.

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
    """Read video stream and store frames in a stack."""

    def __init__(
        self, stack: Stack, preprocess_fn: Callable, camera_view: QLabel
    ) -> None:
        super().__init__()
        self._run_flag = True
        self._stack = stack
        self._preprocess_fn = preprocess_fn
        self._camera_view = camera_view

    def run(self) -> None:
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                qt_img = self._preprocess_fn(cv_img)
                self._camera_view.setPixmap(qt_img)
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                self._stack.put(cv_img)
        # shut down capture system
        cap.release()

    def stop(self) -> None:
        """Sets run flag to False and waits for thread to finish."""
        self._run_flag = False
        self.wait()


class VideoConsumer(QThread):
    """Consume frames from a stack and perform predictions."""

    def __init__(self, stack: Stack, prediction_function: Callable) -> None:
        super().__init__()
        self._run_flag = True
        self._stack = stack
        self._prediction_function = prediction_function

    def run(self) -> None:
        while self._run_flag:
            with self._stack._lock:
                if not self._stack.full():
                    continue

                batch = copy(list(reversed(self._stack.get())))
            self._prediction_function(batch)

    def stop(self) -> None:
        """Sets run flag to False and waits for thread to finish."""
        self._run_flag = False
        self.wait()


# pylint: disable=missing-class-docstring
class MplCanvas(FigureCanvasQTAgg):
    # pylint: disable=unused-argument
    def __init__(self, parent=None, width=5, height=4, dpi=100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class Window(QWidget):
    """Anomaly detection live gui Based on media player code from:

    https://codeloop.org/python-how-to-create-media-player-in-pyqt5/
    """

    def __init__(self, clip_length=16, transforms=None) -> None:
        super().__init__()

        self.camera = None
        self.current_camera_name = None
        self.clip_length = clip_length

        self.frames_stack = Stack(max_size=self.clip_length)

        self.transforms = transforms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.setWindowTitle("Anomaly Media Player")
        self.setGeometry(350, 100, 700, 500)
        self.setWindowIcon(QIcon("player.png"))

        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.init_ui()

        self.y_pred = []

        self.show()

    def init_ui(self) -> None:
        """Create media player object."""
        # setup camera
        self.available_cameras = QCameraInfo.availableCameras()
        self.camera_view = QLabel()
        self.frames_stack = Stack(max_size=self.clip_length)
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
        self.thread = VideoThread(
            stack=self.frames_stack,
            preprocess_fn=self.convert_cv_qt,
            camera_view=self.camera_view,
        )
        self._video_consumer = VideoConsumer(
            stack=self.frames_stack, prediction_function=self.perform_prediction
        )
        self.thread.start()
        self._video_consumer.start()

    def perform_prediction(self, batch):
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
        if len(self.y_pred) > MAX_PREDS:
            del self.y_pred[0]
        self.plot()

    def convert_cv_qt(self, cv_img: np.ndarray) -> QPixmap:
        """Convert from an opencv image to QPixmap."""
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
        """Select camera to display."""
        # getting the selected camera
        self.camera = cv2.VideoCapture(camera)

        # getting current camera name
        self.current_camera_name = self.available_cameras[camera].description()

    def play_video(self) -> None:
        """Change the state of the media player."""
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediastate_changed(self, *_args) -> None:
        """Called when the state of the media player changes."""
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
        ax.set_xlim(0, MAX_PREDS)
        ax.set_ylim(-0.1, 1.1)
        ax.plot(self.y_pred, "*-", linewidth=5)
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
