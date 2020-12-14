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

from bunch import Bunch
from absl import logging

import tensorflow as tf

from tensorpipe.pipe import Funnel


class TestClassificationFunnel(tf.test.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create an Augmentation pipeline !
        config = {
            "batch_size": 1,
            "image_size": [512, 512],
            "transformations": {
                "flip_left_right": None,
                "gridmask": None,
                "random_rotate": None,
            },
            "categorical_encoding": "labelencoder",
        }
        self.config = Bunch(config)
        tf.compat.v1.random.set_random_seed(111111)

    def test_sanity(self):
        funnel = Funnel(
            data_path="testdata", config=self.config, datatype="categorical"
        )
        dataset = funnel.from_dataset(type="train")
        data = next(iter(dataset))
        images = data[0]
        self.assertEqual(self.config.image_size, images[0].shape[:2])


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.test.main()
