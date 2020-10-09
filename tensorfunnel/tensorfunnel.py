import os

import tensorflow as tf
import numpy as np
from logging import logging

from loaders import reader as read
from loders import decoder
from aug import augmenter


class TensorFunnel(read.Reader):
    def __init__(self, data_path, config=None, datatype="bbox", training=True):
        super(TensorFunnel, read.Reader).__init__(config, training=training)
        if not isinstance(data_path, str):
            msg = f"datapath should be str but pass {type(data_path)}."
            logging.error(msg)
            raise TypeError("Only str allowed")

        self._datatype = datatype
        self._data_path = data_path
        self.config = config
        self._training = training
        self.decoder = decoder.Decode(self.config)
        self._tensorrecords_path = self.data_path + "/records/"
        self.augmenter = augmenter.Augmentor(datatype, self.config)

    @property
    def allowed_dataset_types(self):
        return ["categorical", "binary", "bbox"]

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

    def parser(self):
        dataset = tf.data.Dataset.list_files(
            self.tf_path_pattern, shuffle=self._training
        )
        if self._training:
            dataset = dataset.repeat()
        dataset = dataset.interleave(
            self._fetch_records, num_parallel_calls=self.AUTOTUNE
        )

        dataset = dataset.with_options(self.optimized_options)
        if self._training:
            dataset = dataset.shuffle(self.per_shard)

    def encoder(self):
        pass

    def _fetch_records(filename):
        return tf.data.TFRecordDataset(filename).prefetch(1)

    def pretraining(self, ds, cache=False):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        return ds.prefetch(self.AUTOTUNE)

    def dataset(self):
        rawdata = self.parser()
        decode_rawdata = lambda input: self.decoder(
            input
        )  # pylint: enable=g-long-lambda
        dataset = rawdata.map(decode_rawdata, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.prefetch(self.config.batch_size)
        dataset = dataset.batch(
            self.config.batch_size, drop_remainder=self.config.drop_remainder
        )
        dataset = dataset.map(lambda *args: self.augmenter(*args))
        dataset = dataset.map(lambda *args: self.encoder(*args))
        dataset = self.pretraining(dataset)
        return dataset
