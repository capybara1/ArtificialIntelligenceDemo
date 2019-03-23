#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring, invalid-name

"""
GUI that may be used to test models,
trained on classified images
"""

import argparse
import sys

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QIcon, QImage, QPen, QPainter
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction
import numpy as np
import scipy as sp

from ui.sync import SharedData
from ui.algorithms import Classifier

WINDOW_TOP = 100
WINDOW_LEFT = 100
WINDOW_WIDTH = 280
WINDOW_HEIGHT = 280

BRUSH_SIZE = 14
BRUSH_COLOR = Qt.black


class Window(QMainWindow):
    """
    The main window of the application
    """

    def __init__(self, shared_data):
        super().__init__()

        self._shared_data = shared_data

        self.initUI()

        self._drawing = False
        self._last_point = QPoint()

    def initUI(self):

        icon = "icons/pain.png"

        self.setWindowTitle("MNIST Test UI")
        self.setGeometry(WINDOW_TOP, WINDOW_LEFT, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setFixedSize(self.size())
        self.setWindowIcon(QIcon(icon))

        self._image = QImage(self.size(), QImage.Format_Grayscale8)
        self._image.fill(Qt.white)

        main_menu = self.menuBar()
        edit_menu = main_menu.addMenu("Edit")

        clear_action = QAction("Clear", self)
        clear_action.setShortcut("Return")
        clear_action.triggered.connect(self.clear)
        edit_menu.addAction(clear_action)

        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().showMessage("prediction:")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drawing = True
            self._last_point = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self._drawing:
            painter = QPainter(self._image)
            painter.setPen(
                QPen(BRUSH_COLOR, BRUSH_SIZE, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            )
            painter.drawLine(self._last_point, event.pos())
            self._last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drawing = False
            self.evaluateInput()

    def paintEvent(self, _):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self._image, self._image.rect())

    def evaluateInput(self):
        width = self._image.width()
        height = self._image.height()
        ptr = self._image.constBits()
        ptr.setsize(width * height)
        orig_image = np.frombuffer(ptr, dtype=np.uint8).reshape((width, height))
        scaled_image = sp.misc.imresize(orig_image, (28, 28))
        data = np.subtract(
            np.full((28, 28), 255, dtype=scaled_image.dtype), scaled_image
        )
        self._shared_data.provide(data)

    def showResult(self, text: str):
        self.statusBar().showMessage(f"prediction: {text}")

    def clear(self):
        self.statusBar().showMessage("prediction:")
        self._image.fill(Qt.white)
        self.update()


def main():
    parser = argparse.ArgumentParser(description="GUI that may be used to test models.")
    parser.add_argument(
        "--model",
        metavar="PATH",
        type=str,
        required=True,
        help="path to the Tensorflow model file",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    shared_data = SharedData()
    classifier = Classifier(args.model, shared_data)
    classifier.start()
    window = Window(shared_data)
    classifier.classification_completed.connect(window.showResult)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
