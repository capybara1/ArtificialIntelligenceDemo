#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Executable module containing a tool for loading
image data from a zip, providing data as pickled
numpy arrays
"""

from argparse import ArgumentParser
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


@contextmanager
def create_temp_dir():
    """Creates and removes a temporary directory"""
    try:
        path = mkdtemp()
        yield path
    finally:
        rmtree(path)


def process_zip(file_path, categories, shape):
    """Processes the ZIP file"""
    file_name = os.path.basename(file_path)
    file_base_name = os.path.splitext(file_name)[0]
    with ZipFile(file_path, "r") as zip_ref:
        with create_temp_dir() as temp_dir:

            print(f'Extracting "{file_name}"...')
            zip_ref.extractall(temp_dir)

            print("Processing images...")
            train = load_data(temp_dir, categories, shape)

            print("Strong data...")
            random.shuffle(train)
            train_x, train_y = separate_data_and_labels(train)
            write_result(train_x, f"X_{file_base_name}.pickle")
            write_result(train_y, f"Y_{file_base_name}.pickle")

            print("Done")


def load_data(dir_path, categories, shape):
    """Loads and scales data"""
    train = []
    for category in categories:
        print('Processing files from category "{category}"...')
        category_path = os.path.join(dir_path, category)
        class_id = categories.index(category)
        for image_file_name in tqdm(os.listdir(category_path)):
            image_file_path = os.path.join(category_path, image_file_name)
            try:
                img = Image.open(image_file_path)
                img = img.resize(shape[0:2], Image.LANCZOS)
                data = np.array(img)
                train.append([data, class_id])
            except IOError as exception:
                warn(str(exception))
    return train


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


def parse_args():
    """Parses the arguments"""
    parser = ArgumentParser(description="Prepares data in a ZIP file.")
    parser.add_argument("file", metavar="PATH", type=str, help="path to the ZIP file")
    parser.add_argument(
        "--shape",
        metavar="SHAPE",
        type=str,
        required=True,
        help='Shape of the input data e.g "28,28"',
    )
    parser.add_argument(
        "--category", metavar="NAME", type=str, nargs="+", help="Name of a category"
    )
    args = parser.parse_args()

    args.shape = tuple([int(el) for el in args.shape.split(",", 3)])

    if not os.path.exists(args.file):
        raise Exception(f'path "{args.file}" does not refer to an existing file')

    return args


def main():
    """Executes the module"""
    args = parse_args()
    process_zip(args.file, args.category, args.shape)


if __name__ == "__main__":
    main()
