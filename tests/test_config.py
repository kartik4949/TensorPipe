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


class ConfigTest(tf.test.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        config = Bunch(config)
        self.config = config
        tf.compat.v1.random.set_random_seed(111111)

    def test_config_getter(self):
        """Verify config."""
        funnel = Funnel(
            data_path="testdata", config=self.config, datatype="categorical"
        )
        _ = funnel.from_dataset(type="train")
        self.assertEqual(self.config.batch_size, 1)

    def test_config_setter(self):
        """Simple test for config assignment"""
        self.config.batch_size = 2
        self.assertEqual(self.config.batch_size, 2)


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.test.main()
