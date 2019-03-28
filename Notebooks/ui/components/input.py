#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that provides functions used in
the context to pre-process input
"""

import argparse

import numpy as np
import scipy as sp


def parse_args_for_image_input():
    """Parses the arguments for test ui's that accept images as input"""

    parser = argparse.ArgumentParser(description="GUI that may be used to test models.")
    parser.add_argument(
        "--model",
        metavar="PATH",
        type=str,
        required=True,
        help="path to the Tensorflow model file",
    )
    parser.add_argument(
        "--shape",
        metavar="SHAPE",
        type=str,
        required=True,
        help='Shape of the input data e.g "28,28"',
    )
    args = parser.parse_args()
    args.shape = tuple([int(el) for el in args.shape.split(",", 3)])
    return args


def resize_image_data(image_data, desired_shape):
    """ Resize the image data"""
    result = sp.misc.imresize(image_data, desired_shape)
    return result


def invert_image_data(image_data):
    """ Inverts the image data"""
    result = np.subtract(
        np.full(image_data.shape, 255, dtype=image_data.dtype), image_data
    )
    return result


def normalize_image_data(image_data):
    """Normalizes the given image data"""
    result = image_data.astype(float) / 255
    return result


def preprocess_image_data(image_data, shape, invert=False):
    """ Preprocesses the image data"""
    image_data = resize_image_data(image_data, shape)
    if invert:
        image_data = invert_image_data(image_data)
    result = normalize_image_data(image_data)
    return result
