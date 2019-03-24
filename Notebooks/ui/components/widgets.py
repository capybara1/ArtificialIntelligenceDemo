#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring, invalid-name, no-self-use

"""
Moduel providing custom widgets
"""

from PyQt5.QtWidgets import QLineEdit


class FileLocatorEdit(QLineEdit):
    """
    Text input that allows drag and drop of files
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.setPlaceholderText("File Path or URL")
        self.setDragEnabled(True)

    def dragEnterEvent(self, event):
        data = event.mimeData()
        urls = data.urls()
        if urls and urls[0].scheme() == "file":
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        data = event.mimeData()
        urls = data.urls()
        if urls and urls[0].scheme() == "file":
            event.acceptProposedAction()

    def dropEvent(self, event):
        data = event.mimeData()
        urls = data.urls()
        print(data.urls())
        if urls and urls[0].scheme() == "file":
            filepath = str(urls[0].path())[1:]
            self.setText(filepath)
