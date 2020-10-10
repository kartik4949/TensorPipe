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
import math
import inspect

import tensorflow as tf
import tensorflow_addons as tfa

from augment import augmentations


ALLOWED_TRANSFORMATIONS = [
    "flip_left_right",
    "random_rotate",
    "gridmask",
    "random_rotate",
    "random_shear_y",
    "cutout",
    "random_shear_x",
]


def radians(degree):
    pi_on_180 = 0.017453292519943295
    return degree * pi_on_180



class Augmentation:
    def __init__(self, config, transformations, type="bbox"):
        self.config = config
        self.type = type
        self.transformations = transformations
        self._pipeline = []
        self._set_tfa_attrb()
        self.image_size = config.image_size
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
        _ = [
            setattr(self, attrib[0], attrib[1])
            for attrib in inspect.getmembers(tfa.image)
            if inspect.isfunction(attrib[1])
        ]

    @tf.function
    def random_rotate(
        self, image, label, prob=0.6, range=[-25, 25], interpolation="BILINEAR"
    ):
        occur = tf.random.uniform([], 0, 1) < prob
        degree = tf.random.uniform([], range[0], range[1])
        image = tf.cond(
            occur,
            lambda: tfa.image.rotate(
                image, radians(degree), interpolation=interpolation
            ),
            lambda: image,
        )
        return image, label

    @tf.function
    def random_shear_x(self, image, label, prob=0.2, range=[0, 1]):
        occur = tf.random.uniform([], -0.15, 0.15) < prob
        shearx = tf.random.uniform([], range[0], range[1])
        image = tfa.image.shear_x(image, level=shearx, replace=0) if occur else image
        return image, label

    @tf.function
    def random_shear_y(self, image, label, prob=0.2, range=[0, 1]):
        occur = tf.random.uniform([], 0, 1) < prob
        sheary = tf.random.uniform([], range[0], range[1])
        image = tfa.image.shear_y(image, level=sheary) if occur else image
        return image, label

    def gridmask(
        self, image, label, ratio=0.6, rotate=10, gridmask_size_ratio=0.5, fill=1
    ):
        return augmentations.augmentation.get("gridmask")(
            self.image_size,
            ratio=ratio,
            rotate=rotate,
            gridmask_size_ratio=gridmask_size_ratio,
            fill=fill,
        ).__call__(image, label)


class Augment(Augmentation):
    def __init__(self, config, datatype="bbox"):
        self.config = config
        self.transformations = self.config.transformations
        super().__init__(config, self.transformations, type=datatype)
        self.dataset_type = datatype

    def __call__(self, image, label):
        for transform in self._pipeline:
            transform_func, pass_label = transform
            if pass_label:
                image, label = transform_func(image, label)
            else:
                image = transform_func(image)
        return image, label
