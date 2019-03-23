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
import tensorflow as tf
from PyQt5.QtCore import QThread, QMutex, QWaitCondition, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
)
from PIL import Image
import numpy as np

WINDOW_TOP = 100
WINDOW_LEFT = 100
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 70

MODEL_NAME = "tf_mnist_dnn.model"


class SharedData:
    """
    Provides access to shared data
    between Window and Classifier
    """

    def __init__(self):
        self._mutex = QMutex()
        self._data = None
        self._data_available = QWaitCondition()

    def consume(self):
        self._mutex.lock()
        if self._data is None:
            self._data_available.wait(self._mutex)
        result = self._data
        self._data = None
        self._mutex.unlock()
        return result

    def provide(self, data):
        self._mutex.lock()
        self._data = data
        self._data_available.wakeAll()
        self._mutex.unlock()


class Classifier(QThread):
    """
    Implements a classifier for numbers
    using the Tensorflow model internally
    """

    classification_completed = pyqtSignal(str)

    def __init__(self, shared_data):
        super().__init__()
        self._shared_data = shared_data

    def run(self):
        model = tf.keras.models.load_model(MODEL_NAME)
        while True:
            data = self._shared_data.consume()
            predictions = model.predict_classes(data.reshape((1, 28, 28)))
            self.classification_completed.emit(str(predictions[0]))


class Window(QMainWindow):
    """
    The main window of the application
    """

    def __init__(self, shared_data):
        super().__init__()

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

        self._text_input = QLineEdit()
        self._text_input.setPlaceholderText("File Path or URL")
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
            data = loadFromUrl(input_str)
        else:
            data = loadFromFile(input_str)
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
    app = QApplication(sys.argv)
    shared_data = SharedData()
    classifier = Classifier(shared_data)
    classifier.start()
    window = Window(shared_data)
    classifier.classification_completed.connect(window.showResult)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
