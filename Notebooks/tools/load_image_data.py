#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Executable module containing a tool for loading
image data from a zip, providing data as pickled
numpy arrays
"""

from argparse import ArgumentParser, ArgumentTypeError
from contextlib import contextmanager
from tempfile import mkdtemp
from shutil import rmtree
import os
import random
import pickle
from zipfile import ZipFile
from warnings import warn

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


@contextmanager
def create_temp_dir():
    """Creates and removes a temporary directory"""
    try:
        path = mkdtemp()
        yield path
    finally:
        rmtree(path)


def process_zip(file_path, categories, shape, out_path_prefix):
    """Processes the ZIP file"""
    with ZipFile(file_path, "r") as zip_ref:
        with create_temp_dir() as temp_dir:

            print(f'Extracting "{file_path}"...')
            zip_ref.extractall(temp_dir)

            process_dir(temp_dir, categories, shape, out_path_prefix)


def process_dir(dir_path, categories, shape, out_path_prefix):
    """Processes the directory containing the images"""
    encoder = OneHotEncoder(sparse=False)
    encoder.fit([[c.label] for c in categories])

    print("Processing images...")
    train_x, train_y = load_data(dir_path, categories, shape, encoder)

    print("Storing data...")
    write_result(train_x, out_path_prefix + ".x.pickle")
    write_result(train_y, out_path_prefix + ".y.pickle")
    write_result([c.label for c in categories], out_path_prefix + ".l.pickle")

    print("Done")


def normalize(data):
    """Normalizes the given image data"""
    return data.astype(float) / 255


def load_data(dir_path, categories, shape, encoder):
    """Loads and prepares the data"""
    train = []
    for category in categories:
        print(f'Processing files from category "{category.label}"...')
        category_path = os.path.join(dir_path, category.path)
        for image_file_name in tqdm(os.listdir(category_path)):
            image_file_path = os.path.join(category_path, image_file_name)
            try:
                img = Image.open(image_file_path)
                if len(shape) == 3 and shape[2] == 3 and img.mode != "RGB":
                    warn(f'"image_file_path" is not an RGB image')
                    continue
                elif (len(shape) < 3 or shape[2] == 1) and img.mode != "L":
                    img = img.convert("L")
                if not shape is None:
                    img = img.resize(shape[0:2], Image.LANCZOS)
                data = normalize(np.array(img))
                train.append([data, [category.label]])
            except IOError as exception:
                warn(str(exception))

    random.shuffle(train)
    train_x, labels = separate_data_and_labels(train)

    train_x = np.array(train_x)
    train_y = encoder.transform(labels)

    return train_x, train_y


def separate_data_and_labels(train):
    """Separates data and labels"""
    train_x = []
    train_y = []
    for features, label in train:
        train_x.append(features)
        train_y.append(label)
    return train_x, train_y


def write_result(obj, path):
    """Writes the given object to disk"""
    pickle_out = open(path, "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()


class LabelInfo:
    """Helper class to extract information about labels"""

    def __init__(self, arg):
        if not isinstance(arg, str):
            raise ArgumentTypeError("a label must be of type string")
        components = arg.split(":")
        self.label = components[0]
        self.path = components[-1]

    def __repr__(self):
        return f"{self.label}:{self.path}" if self.label != self.path else self.label


def parse_args():
    """Parses the arguments"""
    parser = ArgumentParser(description="Prepares data in a ZIP file.")
    parser.add_argument(
        "in_path",
        metavar="PATH",
        type=str,
        help="path to the directory or the ZIP file containing the images",
    )
    parser.add_argument(
        "categories",
        metavar="LABEL[:PATH]",
        type=LabelInfo,
        nargs="+",
        help="Labels of a category; each label may optionally "
        + "followed by a path if the later differs from the label"
        + 'e.g. "cat:img/cats"',
    )
    parser.add_argument(
        "--shape",
        metavar="SHAPE",
        type=str,
        default=None,
        help='Resizes the images to a uniform shape e.g "28,28"',
    )
    parser.add_argument(
        "--out",
        dest="out_path_prefix",
        metavar="PREFIX",
        type=str,
        default=None,
        help="Prefix used to generate the name of the output file path",
    )
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        raise Exception(
            f'path "{args.in_path}" does not refer to an existing directory or file'
        )

    if not args.shape is None:
        args.shape = tuple([int(el) for el in args.shape.split(",", 3)])

    if args.out_path_prefix is None:
        args.out_path_prefix = os.path.basename(args.in_path)
        if os.path.isfile(args.in_path):
            args.out_path_prefix = os.path.splitext(args.out_path_prefix)[0]

    return args


def main():
    """Executes the module"""
    args = parse_args()
    extension = os.path.splitext(args.in_path)[1]
    if extension.lower() == ".zip":
        process_zip(args.in_path, args.categories, args.shape, args.out_path_prefix)
    else:
        process_dir(args.in_path, args.categories, args.shape, args.out_path_prefix)


if __name__ == "__main__":
    main()
