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

import typeguard

import tensorflow as tf

import wrappers
from register.register import FUNNEL


"""Singleton Design pattern"""


class _singleton(type):

    _max_allowed_instances = 1
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """__call__.

        Args:
            cls: Child class
            args: child class args
            kwargs: child class kwargs
        """
        if len(cls._instances) == cls._max_allowed_instances:
            raise Exception(
                f"{cls.__name__} is allowed to have at most {cls._max_allowed_instances} instances"
            )

        if cls not in cls._instances:
            cls._instances[cls] = super(_singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


def singleton_pattern(cls):
    """singleton_pattern.
    Helper singleton_pattern decorater.
    """
    return _singleton(cls.__name__, cls.__bases__, dict(cls.__dict__))


"""Funnel Interface class"""


@singleton_pattern
class Funnel(object):
    """Funnel.
    Funnel Class which gets the required Funnel given in
    configuration.
    """

    @typeguard.typechecked
    def __new__(
        cls,
        data_path: str,
        config: dict,
        datatype: str = "bbox",
        training: bool = True,
    ):
        # pylint: disable=line-too-long

        """__new__.

        Args:
            data_path: Data path in structured format,please see readme file
                       for more information.
            config: Config passed as dict instance containing all required.
            datatype: Dataset type e.g ['bbox','categorical','segmentation'],
                    bbox - Bounding Box dataset containing object detection
                           data. i.e x1,y1,x2,y2
                    categorical - Categorical data i.e categorical
                                  (multi class) or  binary (two class) for
                                  Classification problems.
            Example:
                    **********************************************************
                    >> from TensorPipe.pipe import Funnel
                    >> funnel = Funnel('testdata',config=config,datatype='categorical')
                    # high performance with parallelism tf.data iterable.
                    >> dataset = funnel.dataset(type = 'train')

        """
        # pylint: enable=line-too-long
        if datatype not in wrappers.ALLOWED_TYPES:
            raise TypeError(
                "datasettype not in ALLOWED_TYPEs, please check\
                             allowed dataset i.e bbox,classification labels,\
                             segmentation."
            )
        _funnel_class = FUNNEL.get(datatype)
        return _funnel_class(
            data_path, config, datatype=datatype, training=training
        )
