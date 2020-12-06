"""
   Copyright 2020 Kartik Sharma

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from abc import ABC, abstractmethod, abstractproperty

import tensorflow as tf
import logging

ALLOWED_TYPES = ["categorical", "binary", "bbox"]


"""Funnel Abstract Class provides essential helper functions across"""


class Funnel(ABC):
    """Funnel.
    Abstract Funnel Class which acts as intterface for three supported
    Class of dataset, and provides helper functions.
    """

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    @property
    def allowed_dataset_types(self):
        return ALLOWED_TYPES

    @property
    @abstractmethod
    def classes(self):
        raise NotImplementedError

    def tf_path_pattern(self, path):
        return os.path.join(path, "*.record")

    @property
    @abstractmethod
    def data_path(self):
        return self._data_path

    @property
    @abstractmethod
    def datatype(self):
        return self._datatype

    @datatype.setter
    def datatype(self, value):
        if value not in self.allowed_dataset_types:
            msg = f"{value} is not in {self.allowed_dataset_types}"
            logging.error(msg)
            raise TypeError("Only str allowed")
        self._data_path = value

    @property
    def size(self):
        return self._size

    @property
    def optimized_options(self):
        options = tf.data.Options()
        options.experimental_deterministic = not self._training
        options.experimental_optimization.map_vectorization.enabled = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        return options

    @abstractmethod
    def parser(self):
        """parser.
        Parser Abstract method which will act as abstract method for
        Base classes.
        """
        raise NotImplementedError(
            "Method parser is not implemented in class " + self.__class__.__name__
        )

    @abstractmethod
    def encoder(self):
        """encoder.
        Encoder Abstract which is abstractmethod, Encoder encodes
        output in required format i.e fixed data size in bbox,segmentation.
        """
        raise NotImplementedError(
            "Method encoder is not implemented in class " + self.__class__.__name__
        )

    def _fetch_records(self, filename):
        """_fetch_records.
                Fetches record files using TfRecordDataset

        Args:
            filename: filename to be fetched
        """
        """_fetch_records.

        Args:
            filename:
        """
        return tf.data.TFRecordDataset(filename).prefetch(1)

    @staticmethod
    def _pad_data(data, pad_value, output_shape):
        """helper function which pads data to given shape."""
        max_instances_per_image = output_shape[0]
        dimension = output_shape[1]
        data = tf.reshape(data, [-1, dimension])
        num_instances = tf.shape(data)[0]
        msg = "ERROR: no. of object are more than max_instances_per_image, please increase max_instances_per_image."
        with tf.control_dependencies(
            [tf.assert_less(num_instances, max_instances_per_image, message=msg)]
        ):
            pad_length = max_instances_per_image - num_instances
        paddings = pad_value * tf.ones([pad_length, dimension])
        padded_data = tf.concat([data, paddings], axis=0)
        padded_data = tf.reshape(padded_data, output_shape)
        return padded_data

    def pad_to_fixed_len(self, *args):
        """
        Bundle inputs into a fixed length.
        """

        image_id, image, bboxes, classes = args
        return (
            image_id,
            image,
            self._pad_data(bboxes, -1, [self.max_instances_per_image, 4]),
            self._pad_data(classes, -1, [self.max_instances_per_image, 1]),
        )

    @staticmethod
    def pretraining(ds, cache=False):
        """pretraining.
                Provides post training configuration i.e prefetching,caching,
                batches,opitmizations.

        Args:
            ds: tf.data dataset reference
            cache: Cache the dataset, WARNING: use only if dataset is small
            enough to fit in ram, default False.
        """
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        return ds.prefetch(tf.data.experimental.AUTOTUNE)

    @abstractmethod
    def from_dataset(self):
        """from_dataset.
        abstractmethod for dataset, returns iterable which can be used
        for feed inputs to neural network.
        provides high performing, low latency data iterable.
        """
        raise NotImplementedError(
            "Method dataset is not implemented in class " + self.__class__.__name__
        )

    @abstractmethod
    def from_tfrecords(self):
        """from_tfrecords.
        abstractmethod for fetch tfrecords, returns iterable which can be used
        for feed inputs to neural network.
        provides high performing, low latency data iterable.
        """
        raise NotImplementedError(
            "Method dataset is not implemented in class " + self.__class__.__name__
        )

    @abstractmethod
    def from_remote(self):
        """from_remote.
        abstractmethod for fetch remote files, returns iterable which can be used
        for feed inputs to neural network.
        provides high performing, low latency data iterable.
        """
        raise NotImplementedError(
            "Method dataset is not implemented in class " + self.__class__.__name__
        )
