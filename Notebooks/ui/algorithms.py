#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring, invalid-name

import tensorflow as tf
from PyQt5.QtCore import QThread, pyqtSignal


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
