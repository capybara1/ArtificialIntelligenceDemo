#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring, invalid-name

"""
GUI that may be used to test models,
trained on classified images
"""

import sys
import os
import re
from io import BytesIO
import pickle

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

        icon = "icons/app.png"

        self.setWindowTitle("Model Test UI")
        self.setGeometry(WINDOW_TOP, WINDOW_LEFT, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setFixedSize(self.size())
        self.setWindowIcon(QIcon(get_absolute_path(icon)))

        widget = QWidget(self)

        predict_button = QPushButton("Predict", widget, enabled=False)
        predict_button.clicked.connect(self.__predictButtonClick)

        self._text_input = FileLocatorEdit(widget)
        self._text_input.textChanged.connect(
            lambda: predict_button.setEnabled(self._text_input.text() != "")
        )
        self._text_input.textChanged.connect(
            lambda: self.statusBar().showMessage("prediction:")
        )
        self._text_input.returnPressed.connect(predict_button.click)

        hbox = QHBoxLayout(widget)
        hbox.addWidget(self._text_input)
        hbox.addWidget(predict_button)

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


def get_absolute_path(relative_path):
    script_dir = os.path.abspath(os.path.dirname(__file__))
    abs_path = os.path.join(script_dir, relative_path)
    return abs_path


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

    labels = []
    if not args.labels_path is None:
        pickle_in = open(args.labels_path, "rb")
        labels = pickle.load(pickle_in)
        pickle_in.close()

    app = QApplication(sys.argv)
    shared_data = SharedData()
    classifier = Classifier(args.model, labels, args.shape, shared_data)
    classifier.start()
    window = Window(args.shape, shared_data)
    classifier.classification_completed.connect(window.showResult)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
