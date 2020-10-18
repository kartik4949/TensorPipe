"""Data Pipeline simple tests."""

from bunch import Bunch
from absl import logging
import tensorflow as tf

from pipe import Funnel
from augment import augment


class AugmentTest(tf.test.TestCase):
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
        config = Bunch(config)

        self.augmentor = augment.Augment(config)
        tf.compat.v1.random.set_random_seed(111111)

    def test_augment_boxes(self):
        """Verify num of boxes are valid and syntax check random four images."""
        images = tf.random.uniform(
            shape=(512, 512, 3), minval=0, maxval=255, dtype=tf.float32
        )
        bboxes = tf.random.uniform(
            shape=(2, 4), minval=1, maxval=511, dtype=tf.int32
        )

        _, bbox = self.augmentor(images, bboxes)
        self.assertEqual(bboxes.shape[0], bbox.shape[0])

    def test_image_dimensions(self):
        images = tf.random.uniform(
            shape=(512, 512, 3), minval=0, maxval=255, dtype=tf.float32
        )
        bboxes = tf.random.uniform(
            shape=(2, 4), minval=1, maxval=511, dtype=tf.int32
        )
        image, bbox = self.augmentor(images, bboxes)
        self.assertEqual(image.shape[1], images.shape[1])


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
        _ = funnel.dataset(type="train")
        self.assertEqual(self.config.batch_size, 1)

    def test_config_setter(self):
        """Simple test for config assignment"""
        self.config.batch_size = 2
        self.assertEqual(self.config.batch_size, 2)


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.compat.v1.disable_eager_execution()
    tf.test.main()
