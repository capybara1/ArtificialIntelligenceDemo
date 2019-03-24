#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring, invalid-name

"""
GUI that may be used to test models,
trained on classified images
"""

import sys
import re
from io import BytesIO

import requests
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QHBoxLayout
from PIL import Image
import numpy as np

from components.algorithms import SharedData, Classifier
from components.input import parse_args_for_image_input, preprocess_image_data
from components.widgets import FileLocatorEdit

WINDOW_TOP = 100
WINDOW_LEFT = 100
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 70


class Window(QMainWindow):
    """
    The main window of the application
    """

    def __init__(self, input_shape, shared_data):
        super().__init__()

        self._input_shape = input_shape
        self._shared_data = shared_data

        self.__initUI()

    def __initUI(self):

        icon = "icons/pain.png"

        self.setWindowTitle("MNIST Test UI")
        self.setGeometry(WINDOW_TOP, WINDOW_LEFT, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setFixedSize(self.size())
        self.setWindowIcon(QIcon(icon))

        predict_button = QPushButton("Predict", enabled=False)
        predict_button.setDefault(True)
        predict_button.clicked.connect(self.__predictButtonClick)

        self._text_input = FileLocatorEdit(self)
        self._text_input.textChanged.connect(
            lambda: predict_button.setEnabled(self._text_input.text() != "")
        )

        hbox = QHBoxLayout()
        hbox.addWidget(self._text_input)
        hbox.addWidget(predict_button)

        widget = QWidget(self)
        widget.setLayout(hbox)

        self.setCentralWidget(widget)

        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().showMessage("prediction:")

    def __predictButtonClick(self, _):
        input_str = str(self._text_input.text())
        if re.match("^https?", input_str, re.IGNORECASE):
            image_data = loadFromUrl(input_str)
        else:
            image_data = loadFromFile(input_str)
        data = preprocess_image_data(image_data, self._input_shape)
        self._shared_data.provide(data)

    def showResult(self, text: str):
        self.statusBar().showMessage(f"prediction: {text}")


def loadFromUrl(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    data = np.array(img)
    return data


def loadFromFile(filePath):
    img = Image.open(filePath)
    data = np.array(img)
    return data


def main():
    args = parse_args_for_image_input()

    app = QApplication(sys.argv)
    shared_data = SharedData()
    classifier = Classifier(args.model, shared_data)
    classifier.start()
    window = Window(args.shape, shared_data)
    classifier.classification_completed.connect(window.showResult)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
