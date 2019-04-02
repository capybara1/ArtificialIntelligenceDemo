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
        "--labels",
        dest="labels_path",
        metavar="PATH",
        type=str,
        help="path to the pickled array of labels",
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


def get_bounding_box(img):
    """Finds the bounding box of the drawing in the image"""
    col_sum = np.sum(img, axis=0).nonzero()
    row_sum = np.sum(img, axis=1).nonzero()
    bb_x1, bb_x2 = col_sum[0][0], col_sum[0][-1]
    bb_y1, bb_y2 = row_sum[0][0], row_sum[0][-1]
    result = ((bb_x1, bb_y1), (bb_x2, bb_y2))
    return result


def crop_image_to_bounding_box(img, bnd_box):
    """Crops the given image to the given bounding box"""
    result = img[bnd_box[0][1] : bnd_box[1][1] + 1, bnd_box[0][0] : bnd_box[1][0] + 1]
    return result


def pad_image(img, shape):
    """Pads the given image"""
    hpad_total = shape[1] - img.shape[1]
    hpad_left = hpad_total // 2
    hpad_right = hpad_total - hpad_left
    hpad = (hpad_left, hpad_right)
    vpad_total = shape[0] - img.shape[0]
    vpad_left = vpad_total // 2
    vpad_right = vpad_total - vpad_left
    vpad = (vpad_left, vpad_right)
    result = np.pad(img, (vpad, hpad), "constant", constant_values=0)
    return result


def preprocess_image_data(image_data, shape, invert=False, center=False, fit=False):
    """Preprocesses the image data"""
    if invert:
        image_data = invert_image_data(image_data)
    if fit or center:
        bnd_box = get_bounding_box(image_data)
        drawing_data = crop_image_to_bounding_box(image_data, bnd_box)
        if fit:
            new_shape = tuple([int(max(drawing_data.shape) * 1.2)] * 2)
        else:
            new_shape = image_data.shape
        image_data = pad_image(drawing_data, new_shape)
        sp.misc.imsave("outfile.jpg", image_data)
    image_data = resize_image_data(image_data, shape)
    result = normalize_image_data(image_data)
    return result
