# TensorPipe
[![Library][tensorflow-shield]][tensorflow-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


![alt text](logo.png)


High Performance Tensorflow Data Pipeline with State of Art Augmentations build as a wrapper aroung tensorflow datasets api, with low level optimizations.

## Requirements

* Python 3.8
* Tensorflow 2.2
* Tensorflow addons
* Sklearn
* typeguard

Install using:

```
pip install tensorflow-addons==0.11.2
pip install tensorflow==2.2.0
pip install sklearn
pip install typeguard
```

## Features

- [x] High Performance tf.data pipline
- [x] Core tensorflow support for high performance
- [x] Classification data support
- [x] Bbox data support
- [ ] Keypoints data support
- [ ] Segmentation data support
- [x] GridMask in core tf2.x
- [x] Mosiac Augmentation in core tf2.x
- [x] CutOut in core tf2.x
- [x] Flexible and easy configuration
- [x] Gin-config support
- [x] Custom numpy function injection.
## Advance Users Section: 
## Example Usage 1
### Create a Data Pipeline for Training. (Categorical Data).
```
from pipe import Funnel                                                         
from bunch import Bunch                                                         
"""                                                                             
Create a Funnel for the Pipeline!                                               
"""                                                                             


# Config for Funnel
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
pipeline = pipeline.from_dataset(type="train")                                       
                                                                                
# Pipline ready to use, iter over it to use.
# Custom loop example.
for data in pipeline:
    image_batch , label_batch = data[0], data[1]
    # you can use _loss = loss(label_batch,model.predict(image_batch))
    # calculate gradients on loss and optimize the model.
    print(image_batch,label_batch)                                      

```

## Example Usage 2
### Create a Data Pipeline for Bbounding Boxes with tf records.

```
import numpy as np
from tensorpipe.pipe import Funnel

"""
Create a Funnel for the Pipeline!
"""

# Custom numpy code for injection.
def numpy_function(image, label):
    """normalize image"""
    image = image / 255.0
    return image, label


config = {
    "batch_size": 2,
    "image_size": [512, 512],
    "transformations": {
        "flip_left_right": None,
        "gridmask": None,
        "random_rotate": None,
    },
    "max_instances_per_image": 100,
    "categorical_encoding": "labelencoder",
    "numpy_function": numpy_function,
}
funnel = Funnel(data_path="tfrecorddata", config=config, datatype="bbox")
dataset = funnel.from_tfrecords(type="train")

for data in dataset:
    print(data[1].shape)

```
# Object Detection Usage
### Now build your custom bounding box funnel with subclassing BboxFunnel.

```
from tensorpipe.funnels import funnels
class CustomObjectDetectionLoader(funnels.BboxFunnel):
      def __init__(self, *args):
          super().__init__(*args)

      def encoder(self,args):
          # encoder is overriden to give custom anchors to the model as per the need.

          image_id, image, bbox, classes = args
          # make custom anchors and encode the image and bboxes as per the model need.
          return image, custom_anchors, classes
      def decoder(self, args):
          # override decoder if using custom tf records and decode your custom tfrecord in this method
          return decoded_data
          
funnel = CustomObjectDetectionLoader(data_path="tfrecorddata", config=config, datatype="bbox")
dataset = funnel.from_tfrecords(type="train")
```

# Steps to build tfrecords and custom input loader.
- use funnels.create_records script to build tfrecord or build your own tfrecords using your custom script.
- if used the create_records script to build records, we dont need to overide decoder, but if using custom
  script overriding the decoder function is mandatory.
- If using anchors in the models, please override encoder with custom anchor script.


## Beginners Section.
## Keras Compatiblity.
### Very simple example to use pipeline with keras model.fit as iterable.
```
import tensorflow as tf
from pipe import Funnel

"""
Create a Funnel for the Pipeline!
"""

config = {
    "batch_size": 2,
    "image_size": [100, 100],
    "transformations": {
        "flip_left_right": None,
        "gridmask": None,
        "random_rotate": None,
    },
    "categorical_encoding": "labelencoder",
}
pipeline = Funnel(data_path="testdata", config=config, datatype="categorical")
# from dataset i.e normal dataset.
pipeline = pipeline.from_dataset(type="train")

# e.g from tfrecords i.e tfrecord dataset.
# pipeline = pipeline.from_tfrecords(type="train") # testdata/train/*.tfrecord

# Create Keras model
model = tf.keras.applications.VGG16(
    include_top=True, weights=None,input_shape=(100,100,3),
    pooling=None, classes=2, classifier_activation='sigmoid'
)

# compile
model.compile(loss='mse', optimizer='adam')

# pass pipeline as iterable
model.fit(pipeline , batch_size=2,steps_per_epoch=5,verbose=1)
```

## Config.
* **image_size** - Output Image Size for the pipeline.
* **batch_size** - Batch size for the pipeline.
* **transformations** - Dictionary of transformations to apply with respective keyword arguments.
* **categorical_encoding** - Encoding for categorical data - ('labelencoder' , 'onehotencoder').

## Augmentations:

### GridMask
Creates a gridmask on input image with rotation defined on range.
* **params**:
    * **ratio** - grid to space ratio
    * **fill** - fill value
    * **rotate** - rotation range in degrees

### MixUp
Mixes two randomly sampled images and their respective labels with given alpha.
* **params**:
    * **alpha** - value for blend function.

### RandomErase
Randomly erases rectangular chunk with is sampled randomly on given image.
* **params**:
    * **prob** - probablity to randomerase on image.

### CutMix
Overlaps a resized randomly sample image on given image with complete overlay on subset portion of image.
* **params**:
    * **prob** - probablity to CutMix on image.

### Mosaic
Creates a mosaic of input 4 images into one single image.
* **params**:
    * **prob** - Probablity to mosaic.

## CutMix, CutOut, MixUp

![alt text](https://www.researchgate.net/publication/340296142/figure/fig1/AS:874996595429376@1585626853032/Comparison-of-our-proposed-Attentive-CutMix-with-Mixup-5-Cutout-1-and-CutMix-3.png)
#### source (https://www.researchgate.net/publication/340296142/figure/fig1/AS:874996595429376@1585626853032/Comparison-of-our-proposed-Attentive-CutMix-with-Mixup-5-Cutout-1-and-CutMix-3.png)

## Mosaic
![alt-text](https://hoya012.github.io/assets/img/yolov4/8.PNG)
#### source (https://hoya012.github.io/assets/img/yolov4/8.PNG)

## Grid Mask
![alt-text](https://storage.googleapis.com/groundai-web-prod/media/users/user_302546/project_404544/images/x1.png)
#### source (https://storage.googleapis.com/groundai-web-prod/media/users/user_302546/project_404544/images/x1.png)


## Release History
* v1.0
    * Bbox, Keypoints, Custom Py Functions Support.(WIP)
* v1.0-beta
    * Classification Support with gridmask and mosaic augmentations.

## Meta

Kartik Sharma – [@linkedIn](https://www.linkedin.com/in/kartik-sharma-aaa021169/) – kartik4949@gmail.com

Distributed under the Apache 2.0 license. See ``LICENSE`` for more information.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[tensorflow-shield]: https://img.shields.io/badge/Tensorflow-2.x-orange
[tensorflow-url]: https://tensorflow.org
[license-shield]: https://img.shields.io/badge/OpenSource-%E2%9D%A4%EF%B8%8F-blue
[license-url]: LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/kartik-sharma-aaa021169/
