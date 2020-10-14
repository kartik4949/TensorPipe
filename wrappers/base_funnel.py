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


from abc import ABC, abstractmethod

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
    def classes(self):
        return self._classes

    @property
    def tf_path_pattern(self):
        return self._tensorrecords_path + "*.tfrecord"

    @property
    def data_path(self):
        return self._data_path

    @property
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
            "Method parser is not implemented in class "
            + self.__class__.__name__
        )

    @abstractmethod
    def encoder(self):
        """encoder.
                Encoder Abstract which is abstractmethod, Encoder encodes
                output in required format i.e fixed data size in bbox,segmentation.
        """
        raise NotImplementedError(
            "Method encoder is not implemented in class "
            + self.__class__.__name__
        )

    def _fetch_records(filename):
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
    def dataset(self):
        """dataset.
                abstractmethod for dataset, returns iterable which can be used
                for feed inputs to neural network.
                provides high performing, low latency data iterable.
        """
        raise NotImplementedError(
            "Method dataset is not implemented in class "
            + self.__class__.__name__
        )
