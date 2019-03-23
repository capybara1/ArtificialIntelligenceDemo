#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring, invalid-name

from PyQt5.QtCore import QMutex, QWaitCondition


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
