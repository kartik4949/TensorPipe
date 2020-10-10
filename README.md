# TensorPipe
![alt text](logo.png)


High Performance Tensorflow Data Pipeline with State of Art Augmentations and low level optimizations.

## Features

- [x] High Performance tf.data pipline
- [x] Core tensorflow support for high performance
- [x] Classification data support
- [ ] Bbox data support
- [ ] Keypoints data support
- [ ] Segmentation data support
- [x] GridMask in core tf2.x
- [x] Mosiac Augmentation in core tf2.x
- [x] CutOut in core tf2.x
- [x] Flexible and easy configuration
- [x] Gin-config support

## Example Usage

```
from pipe import Funnel                                                         
from bunch import Bunch                                                         
"""                                                                             
Create a Funnel for the Pipeline!                                               
"""                                                                             


#All configuration will be avaible soon!!
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
                                                                                
#Pipline ready to use, iter over it to use.                                                      
for data in pipeline:                                                           
    print(data[0].shape, data[1].shape)                                     

```
