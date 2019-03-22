#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GUI that may be used to test models,
trained with the the MNIST dataset.
"""

import tensorflow as tf
from PyQt5.QtCore import Qt, QPoint, QSize, QThread, QMutex, QWaitCondition, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPen, QPainter
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction
import numpy as np
import scipy as sp
import sys

BRUSH_SIZE = 14
BRUSH_COLOR = Qt.black

MODEL_NAME = "tf_mnist_dnn.model"


class SharedData:
    def __init__(self):
        self._mutex = QMutex()
        self._data = None
        self._dataAvailable = QWaitCondition()

    def consume(self):
        self._mutex.lock()
        if self._data == None:
            self._dataAvailable.wait(self._mutex)
        result = self._data
        self._data = None
        self._mutex.unlock()
        return result

    def provide(self, data):
        self._mutex.lock()
        self._data = data
        self._dataAvailable.wakeAll()
        self._mutex.unlock()


class Classifier(QThread):

    classification_completed = pyqtSignal(str)

    def __init__(self, sharedData):
        super().__init__()
        self._sharedData = sharedData

    def run(self):
        model = tf.keras.models.load_model(MODEL_NAME)
        while True:
            data = self._sharedData.consume()
            predictions = model.predict_classes(data)
            self.classification_completed.emit(str(predictions[0]))


class Window(QMainWindow):
    def __init__(self, sharedData):
        super().__init__()

        self._sharedData = sharedData

        self.initUI()

        self._drawing = False
        self._last_point = QPoint()

    def initUI(self):
        top = 100
        left = 100
        width = 280
        height = 280

        icon = "icons/pain.png"

        self.setWindowTitle("MNIST Test UI")
        self.setGeometry(top, left, width, height)
        self.setWindowIcon(QIcon(icon))

        self._image = QImage(self.size(), QImage.Format_Grayscale8)
        self._image.fill(Qt.white)

        mainMenu = self.menuBar()
        editMenu = mainMenu.addMenu("Edit")

        clearAction = QAction("Clear", self)
        clearAction.setShortcut("Return")
        clearAction.triggered.connect(self.clear)
        editMenu.addAction(clearAction)

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

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self._image, self._image.rect())

    def evaluateInput(self):
        width = self._image.width()
        height = self._image.height()
        ptr = self._image.constBits()
        ptr.setsize(width * height)
        orig_image = np.frombuffer(ptr, dtype=np.uint8).reshape((width, height))
        scaled_image = sp.misc.imresize(orig_image, (28, 28))
        data = scaled_image.reshape((1, 28, 28))
        self._sharedData.provide(data)

    def showResult(self, text):
        self.statusBar().showMessage(f"prediction: {text}")

    def clear(self):
        self.statusBar().showMessage("prediction:")
        self._image.fill(Qt.white)
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    sharedData = SharedData()
    classifier = Classifier(sharedData)
    classifier.start()
    window = Window(sharedData)
    classifier.classification_completed.connect(window.showResult)
    window.show()
    app.exec()
