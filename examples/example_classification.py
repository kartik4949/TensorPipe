from wrappers.funnel_wrappers import CategoricalTensorFunnel as Funnel
from pipe import Funnel
from bunch import Bunch
"""
Create a Funnel for the Pipeline!
"""

config = {
    "batch_size": 2,
    "image_size": [512,512],
    "transformations": {
        "flip_left_right": None,
        "gridmask": None,
        "random_rotate":None,
    },
    "categorical_encoding":"labelencoder"
}
config = Bunch(config)
pipeline = Funnel(data_path="testdata", config=config, datatype="categorical")
pipeline = pipeline.dataset(type="train")


for data in pipeline:
    print(data[0].shape)
