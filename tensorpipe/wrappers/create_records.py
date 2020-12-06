""" Create TFrecords from json GTs """
import io
from collections import namedtuple
import os
import logging
import json
from glob import glob

import numpy as np
import cv2
from tqdm import tqdm
from absl import app, flags
import pandas as pd
import tensorflow as tf

from PIL import Image


class CreateRecords:
    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def int64_list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def bytes_list_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def float_list_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def create_tf_example(self, group):
        """create_tf_example.
        create tf example buffer

        Args:
            group: group name
        """
        with tf.io.gfile.GFile(os.path.join(group.filename), "rb") as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)

        width, height = image.size

        filename = group.filename.encode("utf8")
        image_format = b"jpg"
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for _, row in group.bbox.iterrows():
            xmins.append(row["xmin"] / width)
            xmaxs.append(row["xmax"] / width)
            ymins.append(row["ymin"] / height)
            ymaxs.append(row["ymax"] / height)
            if (
                row["xmin"] / width > 1
                or row["ymin"] / height > 1
                or row["xmax"] / width > 1
                or row["ymax"] / height > 1
            ):
                logging.info(row)

            classes_text.append(row["class"].encode("utf8"))
            classes.append(1)
        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/height": self.int64_feature(height),
                    "image/width": self.int64_feature(width),
                    "image/filename": self.bytes_feature(filename),
                    "image/image_id": self.bytes_feature("0".encode("utf8")),
                    "image/encoded": self.bytes_feature(encoded_jpg),
                    "image/format": self.bytes_feature(image_format),
                    "image/bbox/xmin": self.float_list_feature(xmins),
                    "image/bbox/xmax": self.float_list_feature(xmaxs),
                    "image/bbox/ymin": self.float_list_feature(ymins),
                    "image/bbox/ymax": self.float_list_feature(ymaxs),
                    "image/class/text": self.bytes_list_feature(classes_text),
                    "image/class/label": self.int64_list_feature(classes),
                }
            )
        )
        return tf_example
