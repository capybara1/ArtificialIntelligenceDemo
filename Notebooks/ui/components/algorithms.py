#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring, invalid-name

"""
Module that provides classes
related to machine learning algorithms
"""

import tensorflow as tf
from PyQt5.QtCore import QThread, QMutex, QWaitCondition, pyqtSignal


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

    def __init__(self, model_path, shared_data):
        super().__init__()
        self._model_path = model_path
        self._shared_data = shared_data

    def run(self):
        model = tf.keras.models.load_model(self._model_path)
        while True:
            data = self._shared_data.consume()
            predictions = model.predict_classes(data.reshape((1, 28, 28)))
            self.classification_completed.emit(str(predictions[0]))
