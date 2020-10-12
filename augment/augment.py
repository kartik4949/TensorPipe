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

import tensorflow as tf
import tensorflow_addons as tfa

from augment import augmentations
from register.register import AUG

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


def radians(degree: int) -> float:
    """radians.
            helper function converts degrees to radians.
    Args:
        degree: degrees.
    """
    pi_on_180 = 0.017453292519943295
    return degree * pi_on_180


"""Augment Class for interface of Augmentations."""


class Augmentation:
    def __init__(self, config, transformations, type="bbox"):
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

    @tf.function
    def random_rotate(
        self, image, label, prob=0.6, range=[-25, 25], interpolation="BILINEAR"
    ):
        """random_rotate.
                Randomly rotates the given image using rotation range
                and probablity.

        Args:
            image: Image tensor.
            label: label tensor i.e labels,bboxes,keypoints, etc.
            prob: probablity is rotation occurs.
            range: range of rotation in degrees.
            interpolation: interpolation method.

        Example:
            ****************************************************
            image , label = random_rotate(image,label,prob = 1.0)
            visualize(image)
        """
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
        """random_shear_x.
                Randomly shears the given image using shear range
                and probablity in x direction.

        Args:
            image: Image tensor.
            label: label tensor i.e labels,bboxes,keypoints, etc.
            prob: probablity if shear occurs.
            range: range of shear (0,1).

        Example:
            ****************************************************
            image , label = random_shear_x(image,label,prob = 1.0)
            visualize(image)
        """

        occur = tf.random.uniform([], -0.15, 0.15) < prob
        shearx = tf.random.uniform([], range[0], range[1])
        image = (
            tfa.image.shear_x(image, level=shearx, replace=0)
            if occur
            else image
        )
        return image, label

    @tf.function
    def random_shear_y(self, image, label, prob=0.2, range=[0, 1]):
        """random_shear_y.
                Randomly shears the given image using shear range
                and probablity in y direction.

        Args:
            image: Image tensor.
            label: label tensor i.e labels,bboxes,keypoints, etc.
            prob: probablity of shear.
            range: range of shear (0,1).

        Example:
            ****************************************************
            image , label = random_shear_y(image,label,prob = 1.0)
            visualize(image)
        """

        occur = tf.random.uniform([], 0, 1) < prob
        sheary = tf.random.uniform([], range[0], range[1])
        image = tfa.image.shear_y(image, level=sheary) if occur else image
        return image, label

    def gridmask(
        self,
        image,
        label,
        ratio=0.6,
        rotate=10,
        gridmask_size_ratio=0.5,
        fill=1,
    ):
        """gridmask.
                GridMask initializer function which intializes GridMask class.

        Args:
            image: Image tensor.
            label: label tensor i.e labels,bboxes,keypoints, etc.
            ratio: Ratio of grid to space.
            rotate: rotation range for grid.
            gridmask_size_ratio: grid to image_size ratio.
            fill: fill value default 1.
        """
        return AUG.get("gridmask")(
            self.image_size,
            ratio=ratio,
            rotate=rotate,
            gridmask_size_ratio=gridmask_size_ratio,
            fill=fill,
        ).__call__(image, label)


class Augment(Augmentation):
    """Augment.
            Augmentation Interface which performs the augmentation in pipeline
            in sequential manner.
    """

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

    def __call__(
        self, image: tf.Tensor, label: tf.Tensor
    ) -> (tf.Tensor, tf.Tensor):
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
