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
import inspect
import typeguard
from bunch import Bunch
from typing import Optional

import tensorflow as tf
import numpy as np
import logging

from .base_funnel import Funnel
from ..augment import augment
from ..register.register import FUNNEL

__all__ = ["BboxFunnel", "CategoricalTensorFunnel"]

"""Bbox Funnel for bounding box dataset."""


class TFDecoderMixin:
    """Tensorflow decoder."""

    KEYS_TO_FEATURES = {
        "image/encoded": tf.io.FixedLenFeature((), tf.string),
        "image/source_id": tf.io.FixedLenFeature((), tf.string, ""),
        "image/height": tf.io.FixedLenFeature((), tf.int64, -1),
        "image/width": tf.io.FixedLenFeature((), tf.int64, -1),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
        "image/object/area": tf.io.VarLenFeature(tf.float32),
        "image/object/is_crowd": tf.io.VarLenFeature(tf.int64),
    }

    def _decode_image(self, parsed_tensors):
        """Decodes the image"""
        image = tf.io.decode_image(parsed_tensors["image/encoded"], channels=3)
        image.set_shape([None, None, 3])
        return image

    def _decode_boxes(self, parsed_tensors):
        """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
        xmin = parsed_tensors["image/object/bbox/xmin"]
        xmax = parsed_tensors["image/object/bbox/xmax"]
        ymin = parsed_tensors["image/object/bbox/ymin"]
        ymax = parsed_tensors["image/object/bbox/ymax"]
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def decode(self, serialized_example):
        """Decode the serialized example."""
        parsed_tensors = tf.io.parse_single_example(
            serialized_example, self.KEYS_TO_FEATURES
        )
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=""
                    )
                else:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=0
                    )

        image = self._decode_image(parsed_tensors)
        boxes = self._decode_boxes(parsed_tensors)
        decode_image_shape = tf.logical_or(
            tf.equal(parsed_tensors["image/height"], -1),
            tf.equal(parsed_tensors["image/width"], -1),
        )
        image_shape = tf.cast(tf.shape(image), dtype=tf.int64)

        parsed_tensors["image/height"] = tf.where(
            decode_image_shape, image_shape[0], parsed_tensors["image/height"]
        )
        parsed_tensors["image/width"] = tf.where(
            decode_image_shape, image_shape[1], parsed_tensors["image/width"]
        )

        decoded_tensors = {
            "image": image,
            "height": parsed_tensors["image/height"],
            "width": parsed_tensors["image/width"],
            "groundtruth_classes": parsed_tensors["image/object/class/label"],
            "groundtruth_boxes": boxes,
        }
        return decoded_tensors


@FUNNEL.register_module(name="bbox")
class BboxFunnel(Funnel, TFDecoderMixin):
    """BboxFunnel.
    BboxFunnel Class for Bbox dataset,This class will provide
    data iterable with images,bboxs or images,targets with required
    augmentations.
    """

    def __init__(
        self,
        data_path: str,
        config: Optional[dict] = None,
        datatype="bbox",
        training=True,
    ):
        """__init__.

        Args:
            data_path: Dataset Path ,this is required in proper structure
            please see readme file for more details on structuring.
            config: Config File for setting the required configuration of datapipeline.
            training:Traning mode on or not?

        Example::
            e.g 1
            >> funnel = Funnel(config=config, datatype = "bbox")
            >> data = next(iter(funnel.from_tfrecords('tfrecord_data/' , type="train")))

            e.g 2:
            class CustomFunnel(BboxFunnel):
                def __init__(self, *args):
                    super().__init__(*args)

                def encoder(self,args):
                    # should be overriden if there is a need for anchors in the model.

                    image_id, image, bbox, classes = args
                    # make custom anchors and encode the image and bboxes as per
                    /the model need.
                    return image, custom_anchors, classes
            funnel = CustomFunnel()



        """
        # bunch the config dict.
        config = Bunch(config)
        if not isinstance(data_path, str):
            msg = f"datapath should be str but pass {type(data_path)}."
            logging.error(msg)
            raise TypeError("Only str allowed")
        if not os.path.exists(data_path):
            msg = f"path doesnt exists"
            logging.error(msg)
            raise TypeError("Path doesnt exists")

        self._datatype = "bbox"
        self._data_path = data_path
        self.config = config
        self._training = training
        self._drop_remainder = self.config.get("drop_remainder", True)
        self.augmenter = augment.Augment(self.config, datatype)
        self.numpy_function = self.config.get("numpy_function", None)
        self._per_shard = self.config.get("shard", 10)  # hardcoded shard size
        self.max_instances_per_image = self.config.get("max_instances_per_image", 100)
        self.numpy_function = self.config.get("numpy_function", None)

        if self.numpy_function:
            assert callable(self.numpy_function), "numpy_function should be a callable."
            assert len(
                inspect.getfullargspec(self.numpy_function).args
            ), "py_function should be having two arguments."

    @property
    def datatype(self):
        return self._datatype

    @property
    def classes(self):
        return self._classes

    def parser(self, dataset_folder):
        """parser for reading images and bbox from tensor records."""
        dataset = tf.data.Dataset.list_files(
            self.tf_path_pattern(os.path.join(self.data_path, dataset_folder)),
            shuffle=self._training,
        )
        if self._training:
            dataset = dataset.repeat()
        dataset = dataset.interleave(
            self._fetch_records, num_parallel_calls=self.AUTOTUNE
        )

        dataset = dataset.with_options(self.optimized_options)
        if self._training:
            dataset = dataset.shuffle(self._per_shard)
        return dataset

    def encoder(self):
        """Method expected to be overidden by the user. """
        raise NotImplementedError()

    def decoder(self, value):
        """helper decoder, a wrapper around tfrecorde decoder."""
        data = self.decode(value)
        image_id = 1.0
        image = data["image"]
        boxes = data["groundtruth_boxes"]
        classes = data["groundtruth_classes"]
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        return (image_id, image, boxes, classes)

    def from_tfrecords(self, type="train"):
        """tf_records.
        Returns a iterable tf.data dataset ,which is configured
        with the config file passed with require augmentations.
        """
        dataset = self.parser(type)
        decode_rawdata = lambda input_records: self.decoder(
            input_records
        )  # pylint: enable=g-long-lambda
        dataset = dataset.map(decode_rawdata, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.prefetch(self.config.batch_size)

        # custom numpy function to inject in datapipeline.
        def _numpy_function(img_id, img, bbox, classes):
            _output = tf.numpy_function(
                func=self.numpy_function,
                inp=[img, bbox],
                Tout=(tf.float32, tf.float32),
            )
            return img_id, _output[0], _output[1], classes

        if self._training:
            dataset = dataset.map(
                lambda image_id, image, bbox, classes: self.augmenter(
                    image, bbox, image_id, classes, return_image_label=False
                )
            )
            if self.numpy_function:
                dataset = dataset.map(_numpy_function, num_parallel_calls=self.AUTOTUNE)

        # try if encoder is implemented.
        try:
            self.encoder()
        except NotImplementedError:
            logging.info("Encoder is not implemented,giving raw output.")
        else:
            dataset = dataset.map(lambda *args: self.encoder(*args))

        # pad to fixed length.
        dataset = dataset.map(
            self.pad_to_fixed_len,
            num_parallel_calls=self.AUTOTUNE,
        )

        # make batches.
        dataset = dataset.batch(
            self.config.batch_size, drop_remainder=self._drop_remainder
        )
        dataset = self.pretraining(dataset)
        return dataset

    @property
    def data_path(self):
        return self._data_path

    def from_dataset(self, tfrecord_path: str = None):
        # TODO(kartik4949) : write me
        # fetch raw data
        raise NotImplementedError

    def from_remote(self, remote_path: str = None):
        # TODO(kartik4949) : write me
        # fetch remote files
        raise NotImplementedError


@FUNNEL.register_module(name="categorical")
class CategoricalTensorFunnel(Funnel):
    # pylint: disable=line-too-long
    """CategoricalTensorFunnel
        TensorFunnel for Categorical Data provides DataPipeline according
        to config passed with required augmentations.

    Example: ***********************************************
            funnel = CategoricalTensorFunnel('testdata', config=config, datatype='categorical')
            iterable = funnel.dataset(type = 'train')

    Note: This class can only be used for categorical dataset.
          i.e either multiclass or binary.
    """
    # pylint: enable=line-too-long

    @typeguard.typechecked
    def __init__(
        self,
        data_path: str,
        config: Optional[dict] = None,
        datatype="categorical",
        training=True,
    ):
        """__init__.

        Args:
            data_path: Dataset Path which should be in structure way
            please see readme file for more details on structuring.
            config: Config file , a dict file contains all required attributes
            to configure.
            datatype: Dataset Type i.e (Bbox , Labels ,Segmentations)
            bbox dataset is object detection dataset which will be provided in
            form of [image,bboxs] or [image, class_targets,bbox_targets].
            training: is pipeline in training mode or not?
        """

        # bunch the config dict.
        config = Bunch(config)
        if not isinstance(data_path, str):
            msg = f"datapath should be str but pass {type(data_path)}."
            logging.error(msg)
            raise TypeError("Only str allowed")

        self._datatype = datatype
        self._data_path = data_path
        self.config = config
        self._training = training
        self._shuffle_buffer = None
        self._batch_size = self.config.get("batch_size", 32)
        self._image_size = self.config.get("image_size", [512, 512])
        self._drop_remainder = self.config.get("drop_remainder", True)
        self.augmenter = augment.Augment(self.config, datatype)
        self.numpy_function = self.config.get("numpy_function", None)

        if self.numpy_function:
            assert callable(self.numpy_function), "numpy_function should be a callable."
            assert len(
                inspect.getfullargspec(self.numpy_function).args
            ), "py_function should be having two arguments."

    def categorical_encoding(self, labels):
        """categorical_encoding.
                Encodes the labels with given encoding in config file.

        Args:
            labels: Labels to encode
        """
        encoding = (
            self.config.categorical_encoding
            if self.config.categorical_encoding
            else "onehot"
        )
        if encoding == "onehot":
            from sklearn.preprocessing import (
                OneHotEncoder,
            )  # pylint: disable=g-import-not-at-top

            encoding = OneHotEncoder(drop="if_binary", sparse=False)
        else:
            from sklearn.preprocessing import (
                LabelEncoder,
            )  # pylint: disable=g-import-not-at-top

            encoding = LabelEncoder()
        labels = encoding.fit_transform(labels)
        return labels

    @property
    def get_id_to_imagefile(self):
        return self._get_id_to_imagefile

    @property
    def classes(self):
        return self._classes

    @property
    def data_path(self):
        return self._data_path

    @property
    def datatype(self):
        return self._datatype

    def resize(self, image):
        return tf.image.resize(
            image,
            self._image_size,
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=False,
            antialias=False,
            name=None,
        )

    @get_id_to_imagefile.setter
    def get_id_to_imagefile(self, value):
        if not isinstance(value, dict):
            msg = "Only dict assign is allowed"
            logging.error(msg)
            raise TypeError(msg)
        self._get_id_to_imagefile = value

    def _generate_ids(self, image_files):
        """_generate_ids.
                Generate igs for the imagefiles, which will be further used
                to parse image file to read.

        Args:
            image_files: images files list containing filename of images.
        """
        # TODO: (HIGH) make get_id_to_imagefile as dataclass.
        self._get_id_to_imagefile = {}
        _ = [
            self._get_id_to_imagefile.update({id: image_file})
            for id, image_file in enumerate(image_files)
        ]
        return list(self._get_id_to_imagefile.keys())

    def _get_file_labels(self, subset):
        """_get_file_labels.
        returns files, labels which will be further used for reading images.
        """
        _images = []
        _labels = []

        for label_folder in os.listdir(self.data_path + "/" + subset):
            for images in os.listdir(
                self.data_path + "/" + subset + "/" + label_folder
            ):
                _images.append(
                    self.data_path + "/" + subset + "/" + label_folder + "/" + images
                )
                _labels.append(label_folder)

        self._classes = set(_labels)
        _labels = np.reshape(np.asarray(_labels), (-1, 1))
        _labels = self.categorical_encoding(_labels)
        _labels = np.reshape(np.asarray(_labels), (-1, 1))
        self._size = len(_images)
        assert len(_images) == len(_labels), "Length of Images and Labels didnt match"
        return _images, _labels

    @typeguard.typechecked
    def parser(self, subset: str) -> tf.data:
        """parser for reading images and bbox from tensor records."""
        dataset = tf.data.Dataset.from_tensor_slices(self._get_file_labels(subset))

        if self._training:
            dataset = dataset.shuffle(self._size)
            dataset = dataset.repeat()

        dataset = dataset.map(self._read, num_parallel_calls=self.AUTOTUNE)
        return dataset

    @tf.function
    def _read(self, image, label):
        """Tensorflow Read Image helper function."""
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, try_recover_truncated=True)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = self.resize(image)
        return image, label

    def encoder(self, dataset):
        return dataset

    @typeguard.typechecked
    def from_dataset(self, type: str = "train") -> tf.data:
        """dataset.
                Dataset function which provides high performance tf.data
                iterable, which gives tuple comprising (x - image, y - labels)
                Iterate over the provided iterable to for feeding into custom
                training loop for pass it to keras model.fit.

        Args:
            type: Subset data for the current dataset i.e train,val,test.
        """
        if type.lower() not in ["train", "val", "test", "validation"]:
            raise Exception("Subset Data you asked is not a valid portion")

        dataset = self.parser(type)
        dataset = dataset.prefetch(self._batch_size)

        # custom numpy function to inject in datapipeline.
        def _numpy_function(img, lbl):
            _output = tf.numpy_function(
                func=self.numpy_function,
                inp=[img, lbl],
                Tout=(tf.float32, tf.int64),
            )
            return _output[0], _output[1]

        if self._training:
            dataset = dataset.map(self.augmenter, num_parallel_calls=self.AUTOTUNE)
            if self.numpy_function:
                dataset = dataset.map(_numpy_function, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.batch(self._batch_size, drop_remainder=self._drop_remainder)
        dataset = self.pretraining(dataset)
        return dataset

    def from_tfrecords(self, tfrecord_path: str = None):
        # TODO(kartik4949) : write me
        # fetch tf_records
        raise NotImplementedError

    def from_remote(self, remote_path: str = None):
        # TODO(kartik4949) : write me
        # fetch remote files
        raise NotImplementedError
