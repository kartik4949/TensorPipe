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

import functools
import inspect
import typeguard
from typing import List

import tensorflow as tf
import tensorflow_addons as tfa

from ..augment import augmentations
from ..register.register import AUG

ALLOWED_TRANSFORMATIONS = [
    "flip_left_right",
    "random_rotate",
    "gridmask",
    "random_rotate",
    "random_shear_y",
    "cutout",
    "mosaic",
    "random_shear_x",
]


"""Augment Class for interface of Augmentations."""


class Augmentation(augmentations.TransformMixin):
    """Augmentation.
    Class Augmentation which consists inhouse augmentations and builds
    the transformations pipeline with given transformations in config.
    ::

    Example:

        augment = Augmentation(config,["random_rotate","gridmask"])
        # Use the pipeline and iterate over function in the pipeline.
        pipeline = augment._pipeline

    """

    @typeguard.typechecked
    def __init__(self, config: dict, transformations: dict, type: str = "bbox"):
        """__init__.
                Augmentation class provides and builds the augmentations pipe-
                line required for tf.data iterable.

        Args:
            config: config file containing augmentations and kwargs required.
            transformations: transformations contains list of augmentations
            to build the pipeline one.
            type: type of dataset to built the pipeline for e.g bbox,
            keypoints,categorical,etc.
        """
        self.config = config
        self.type = type
        self.transformations = transformations
        self._pipeline = []
        self._set_tfa_attrb()
        self.image_size = config.image_size
        # builds the augment pipeline.

        for transform, kwargs in transformations.items():
            if transform not in ALLOWED_TRANSFORMATIONS and not hasattr(
                tf.image, transform
            ):
                raise ValueError(
                    f"{transform} is not a valid augmentation for \
                            tf.image or TensorPipe,please visit readme section"
                )

            kwargs = kwargs if isinstance(kwargs, dict) else {}

            if hasattr(tf.image, transform):
                transform = getattr(tf.image, transform)
                transform = functools.partial(transform, **kwargs)
                self._pipeline.append((transform, False))
            else:
                transform = getattr(self, transform)
                transform = functools.partial(transform, **kwargs)
                self._pipeline.append((transform, True))

    def _set_tfa_attrb(self):
        """_set_tfa_attrb.
        helper function which bounds attributes of tfa.image to self.
        """
        _ = [
            setattr(self, attrib[0], attrib[1])
            for attrib in inspect.getmembers(tfa.image)
            if inspect.isfunction(attrib[1])
        ]


class Augment(Augmentation):
    """Augment.
    Augmentation Interface which performs the augmentation in pipeline
    in sequential manner.
    """

    @typeguard.typechecked
    def __init__(self, config: dict, datatype: str = "bbox"):
        """__init__.

        Args:
            config: config file.
            datatype: dataset type i.e bbox,keypoints,caetgorical,etc.
        """
        self.config = config
        self.transformations = self.config.transformations
        super().__init__(config, self.transformations, type=datatype)
        self.dataset_type = datatype

    @typeguard.typechecked
    def __call__(self, image: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """__call__.
                Callable which is invoked in tfdata pipeline and performs the
                actual transformation on images and labels.

        Args:
            image: Image Tensor tf.Tensor.
            label: Label tensor tf.Tensor.

        Returns:
            Returns the transform image and labels.
        """
        for transform in self._pipeline:
            transform_func, pass_label = transform
            if pass_label:
                image, label = transform_func(image, label)
            else:
                image = transform_func(image)
        return image, label
