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

import tensorflow as tf

import wrappers
from register.register import FUNNEL

"""Funnel Interface class"""


class Funnel(object):
    """Funnel.
            Funnel Class which gets the required Funnel given in
            configuration.
    """

    def __init__(self, data_path, config, datatype="bbox"):
        # pylint: disable=line-too-long

        """__init__.

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
                    from TensorPipe.pipe import Funnel

                    funnel = Funnel('testdata',config=config,datatype='categorical')

                    # high performance with parallelism tf.data iterable.
                    dataset = funnel.dataset(type = 'train')

                    for data in dataset:
                        # feed the data to NN or visualize.
                        print(data[0])
        """
        # pylint: enable=line-too-long
        if datatype not in wrappers.ALLOWED_TYPES:
            raise TypeError(
                "datasettype not in ALLOWED_TYPEs, please check\
                             allowed dataset i.e bbox,classification labels,\
                             segmentation."
            )
        _funnel_class = FUNNEL.get(datatype)
        self._funnel = _funnel_class(data_path, config, datatype)

    def dataset(self, type: str = "Train") -> tf.data:
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
        # high performance tf.data iterable
        iterable = self._funnel.dataset(type)
        return iterable
