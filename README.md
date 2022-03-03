# ArcFace_MATLAB
<!-- This is the "Title of the contribution" that was approved during the Community Contribution Review Process --> 

[![View Implementation-ArcFace-in-MATLAB on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://jp.mathworks.com/matlabcentral/fileexchange/107490-implementation-arcface-in-matlab)

This is implementatoin of ArcFace(https://arxiv.org/abs/1801.07698) in MATLAB.
In this demo, train and test pet images by using ArcFace.
This network can 
1. calculate the similarity between your pet images and each breed of dogs and cats.
2. calculate the similarity between your pet images and the other pet images.

# Products used in development

- MATLAB R2021b
- Image Processing Toolbox
- Deep Learning Toolbox
- add on: Deep Learning Toolbox Model for ResNet-50 Network
- Parallel Computing Toolbox (if you want to use GPU in training)


## Setup
dataset: 
Datasets is automatically downloaded from Oxford-IIIT Pet Datset(https://www.robots.ox.ac.uk/~vgg/data/pets/).

To Run:  
・Train and Test:(GPU is recommanded)  
　1．Open arcface_train.mlx  
　2. Change directory to "arcface_demo"  
　3. Execute Run   
・Test:  
　1．Open arcface_train.mlx  
　2. Change directory to "arcface_demo"  
　3. Go to "Test Model" Section  
　　 If you want to use train model by run, remove commnet of "load dlnet_end.mat"  
　4．Run section of "Test Model"  

## Getting Started 
Please open arcface_train.mlx　and read instructions

## Examples
To learn how to use this in testing workflows, see arface_train.mlx. 


## License
The license for Implementation-ArcFace-in-MATLAB is available in the [LICENSE.TXT](LICENSE.TXT) file in this GitHub repository.

Include any other License information here, including third-party content using separate license agreements 

## Community Support
[MATLAB Central](https://www.mathworks.com/matlabcentral)

Copyright (c) 2022 Nobuaki Fukuchi.