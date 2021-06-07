# Introduction

This is an unofficial inplementation of [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/abs/1711.06396) in TensorFlow. A large part of this project is based on the work [here](https://github.com/jeasinema/VoxelNet-tensorflow). Thanks to [@jeasinema](https://github.com/jeasinema). This work is a modified version with bugs fixed and better experimental settings to chase the results reported in the paper (still ongoing).

This repo converted voxelnet(Single Object Detection) into Multi-label 3D object detection(mainly Pedestrian and Cyclist)  with Lidar point cloud and model is trained and completely analysed for different distance and classes at the same time on different data sets, such as Kitti and waymo. Initially I tried to annotate the custom data sets as well, but currently there are no user friendly and well developed open source tools available.   

# Dependencies
- `python3.5+`
- `TensorFlow` (tested on 1.4.1)
- `opencv`
- `shapely`
- `numba`
- `easydict`
- `mayavi`  - for visualization of results
- `Open3D`

- Please install the dependecies based on request.

# Installation
1. Clone this repository.
2. Compile the Cython module
```bash
$ python3 setup.py build_ext --inplace
```
3. Compile the evaluation code
```bash
$ cd kitti_eval
$ g++ -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp
```
4. grant the execution permission to evaluation script
```bash
$ cd kitti_eval
$ chmod +x launch_test.sh
```

# Data Preparation
1. Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Data to download include:
    * Velodyne point clouds (29 GB): input data to VoxelNet
    * Training labels of object data set (5 MB): input label to VoxelNet
    * Camera calibration matrices of object data set (16 MB): for visualization of predictions
    * Left color images of object data set (12 GB): for visualization of predictions

2. In this project, we use the cropped point cloud data for training and validation. Point clouds outside the image coordinates are removed. Update the directories in `data/crop.py` and run `data/crop.py` to generate cropped data. Note that cropped point cloud data will overwrite raw point cloud data.

2. Split the training set into training and validation set according to the protocol [here](https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz). And rearrange the folders to have the following structure:
```plain
└── DATA_DIR
       ├── training   <-- training data
       |   ├── image_2
       |   ├── label_1
       |   └── velodyne
       └── validation  <--- validation data
       |   ├── image_2
       |   ├── label_1
       |   └── velodyne
       └── testing  <--- evaluation data
       |   ├── image_2
       |   ├── label_1
       |   └── velodyne
```
        
3. Update the dataset directory in `config.py` and `kitti_eval/launch_test.sh`. Please modify the line number 787 in file evaluate_object_3d_offline.cpp

# Train
1. Specify the GPUs to use in `config.py`
2. run `train.py` with desired hyper-parameters to start training:
```bash
$ python3 train.py --alpha 1 --beta 10
```
Note that the hyper-parameter settings introduced in the paper are not able to produce high quality results. So, a different setting is specified here.

Training on two Nvidia 1080 Ti GPUs takes around 3 days (160 epochs as reported in the paper). During training, training statistics are recorded in `log/default`, which can be monitored by tensorboard. And models are saved in `save_model/default`. Intermediate validation results will be dumped into the folder `predictions/XXX/data` with `XXX` as the epoch number. And metrics will be calculated and saved in  `predictions/XXX/log`. If the `--vis` flag is set to be `True`, visualizations of intermediate results will be dumped in the folder `predictions/XXX/vis`.

3. When the training is done, executing `parse_log.py` will generate the learning curve.
```bash
$ python3 parse_log.py predictions
```

4. There is a pre-trained model for car in `save_model/pre_trained_pedestrian_cyclist`.


# Evaluate
1. run `test.py -n default` to produce final predictions on the validation set after training is done. Change `-n` flag to `pre_trained_car` will start testing for the pre-trained model (only car model provided for now).
```bash
$ python3 test.py
```
results will be dumped into `predictions/data`. Set the `--vis` flag to True if dumping visualizations and they will be saved into `predictions/vis`.

2. run the following command to measure quantitative performances of predictions:
```bash
$ ./kitti_eval/evaluate_object_3d_offline [DATA_DIR]/validation/label_1 ./predictions
```

# Changes made to the architecture for multi-label object detection
 Please see the comments in the files in model folder for more info.
![rpn](https://user-images.githubusercontent.com/64356491/120967940-7944a480-c768-11eb-8975-162c42e7ef83.png)



# Voxelization
Implementation of Preprocessing code(voxelization). But in order to use voxelnet based network on Jetson or embbeded devices, we need to implement voxelization on CUDA.
Recently, tensorflow team introduce tf3d, where they used tensorflow framework to write preprocessing setup, which directly runs on gpu.
   
# Performances

The current implementation and training scheme are able to produce results in the tables below. Pedestrian and cyclist are consider at the same time. 

##### Pedestrian detection performance: AP on KITTI test set

| Pedestrian | Easy | Moderate | Hard |
|:-:|:-:|:-:|:-:|
| BEV | 52.0   | 51.30  | 47.50 |
| 3D  | 42.45  | 42.11  | 39.45 |

##### Cyclist  detection performance: AP on KITTI test set

| Cyclist | Easy | Moderate | Hard |
|:-:|:-:|:-:|:-:|
| BEV | 53.87  | 53.13 | 50.34 |
| 3D  | 43.16  | 43.36 | 42.89 |

# TODO
- [] improve the performance by using 3D sparse Convolution instead of Normal 3D convolution
- [] Add the fusion code
- [] Add ROI pooling layer
- [] Add waymo results
- [] A brief description about tackling of class imbalance in 3D/2D
- [] Add visual representation for Multi-Label voxelnet
- [] Convert the code into tf2 from tf1


# Important Observation 1
According to literature review and my practical experience, below points to increase the performance of the model,I did some of them myself and still doing rest
- Use of 3D sparse convolution layer as a middle layer
- Use of more balanced data with more object instances
- More training 
- Use of ROI pooling after the proposal generation, ROI pooling only on certain proposals
- Point cloud registration results in more point cloud density and more accuracy especially upto below 50 m
- Lastly, use of late fusion using VGG or resnet with camera

# Important Observation 2
##### Voxelnet
- Voxelnet Based networks need an optimization of vertical scene size for each data sets
- Unlike pointpillars,height in the scene size is an hyper parameters depending upon datasets
- Give better accuracy than any lidar only or lidar + image based networks
- No loss of information like in complex yolo, MV3D and AVOD, so better accuracy

##### Pointnet
- Good for extraction of local and neighbouring features, whcih results in better localization accuray
- But more inference time, especially in pre processing step
- Need to be tested how the network behaves when the point cloud density is too large


##### Note
Doing all the above mentioned things, I am pretty sure that, accuracy might reach upto 80% on both pedestrian and cyclist. Since master thesis is for limited time, I can not do everything. Hope somebody can make use of everything. In the meantime, I will try to implement all the above mentioned points and update the repo, whenever I get time.

