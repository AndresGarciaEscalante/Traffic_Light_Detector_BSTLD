# Traffic Light Object Detection using Bosch Small Traffic Light Dataset

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!--[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]-->



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="gif/efficientDet.gif" alt="Logo" width="600" height="600">
  </a>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#setup-the-environment">Setup the Environment</a></li>
        <li><a href="#scripts"> Execute Scripts</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>

  </ol>
</details>



<!-- ABOUT THE PROJECT -->
# About The Project

<!--[![Product Name Screen Shot][product-screenshot]](https://example.com -->

Traffic lights are control flow devices that guarantee security for drivers. Due to the lack of concentration from the drivers, which might cause car accidents. Therefore, this project aims to contribute to with the detection an classificaction of traffic lights using state-of-the-art Convolutional Neural Network. 

This project is the first part of my master thesis from Tecnologico de Monterrey, the name of the thesis is ***Emergency Brake Driver Assistant System from Traffic Lights using Deep Learning and Fuzzy Logic.***

***Important:*** The [Bosch Small Traffic Light Dataset](https://github.com/bosch-ros-pkg/bstld) was used as a start point for the project.

## Built with
### Environment
The project was built using:
* Tensorflow 2.3.1
* CUDA 10.1
* cuDNN 7.6
* Python 3.6.8
* Ubuntu 16.04
* Docker 18.09.2

***Important:*** The Environment configuration was setup based on the criteria of [Compatibility of Tensorflow Linux GPU](https://www.tensorflow.org/install/source#tested_build_configurations)

# Getting Started
## Setup the Environment
### 1. Docker image
For this project, I used the docker images provided by the [NGC](https://ngc.nvidia.com/catalog). The docker image that I used for the project is [Tensorflow](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow/tags) and look for the tag ***19.07-py3***: 

```
docker pull nvcr.io/nvidia/tensorflow:19.07-py3
```

Once the image is pulled, then you can run the container. The command that I used to execute the container is:

```
nvidia-docker run --name TL_OD_V4 -v /raid/home/A01375067/Traffic_Light_Detector_BSTLD:/workspace -it -p 8888:8888 nvcr.io/nvidia/tensorflow:19.07-py3
```

***Important:*** You can change the name of the container, the path of the files that will be included in the docker container, and the port.

### 2. Install dependencies
Once you are inside the container, execute the ```setup.py``` to install all the dependencies:

```
python setup.py
```
## Scripts 
### 3. Scripts that provide information of the dataset 
To provide information of the dataset. Make sure to use the following commands:

Number of Annotations, images, classes, size of images, and more:

```
python dataset_stats.py input_yaml
```

***Important:*** Instead of the ```input_yaml``` provide the path of the yaml files that will be evaluated

Display the ground truth in a video:
```
python write_label_images input.yaml [output_folder]
```

***Important:*** Instead of the ```input_yaml``` provide the path of the yaml files that will be evaluated, and provide the path where the sequence of images will be stored ```[output_folder]```

### 4. Scripts for training a CNN Model
The following scripts are a must to train a CNN model, please execute them in order.

#### 4.1 Dataset to TFrecords
To convert the dataset to TFrecords execute the following command:
```
python to_tfrecords.py --train_yaml ../label_files/train.yaml --test_yaml ../label_files/test.yaml --additional_yaml ../label_files/additional_train.yaml --dataset_folder ../label_files
```

After the script is executed, 3 tfrecords will be created ```train.tfrecord, test.tfrecord, valid.tfrecord```.

***Important:*** The script was modified to not suffle the images of the tfrecords.

#### 4.2 Pretrained Model
Before start training a model, make sure to download a pretrainted from [tensorflow2_pretrained_models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Once you have selected the desired model you can downloaded executing the following command:
```
wget link_to_the_pretrained_model
```

To unzip the model use the following command:
```
tar -xvzf replace.tar.gz
```

***Important:*** The commands shown above are only used to test new models. If you want to test the model I trained please omit this step.

#### 4.3 Config Files
There are several ***config files*** that are supported by [tensorflow2_config files](https://github.com/tensorflow/models/tree/master/research/object_detection/configs/tf2). 

***Important:*** Do not forget to modify the paths from the ```train and valid tfrecords```, the ```.pbtxt``` file, and the path from the ***pretrained model***. Use the ***config files*** provided in the repo as a guide for filling this paths.

#### 4.4 Train the model 
For this section, we need to use three terminals. The first one is for training the model and it uses the ```train.tfrecord```, the second terminal is for validation and it uses the ```valid.tfrecord```. The last terminal is to keep track of the ***losses, mAP, precision, recall, learning rate***. Please execute the following commands at the same time:

#### Terminal 1
```
CUDA_VISIBLE_DEVICES=6 python model_main_tf2.py --model_dir=/workspace/tf_object_detection/training/experiments/models4 --pipeline_config_path=/workspace/tf_object_detection/training/experiments/models4/ssd_efficient_1024.config
```
#### Terminal 2
```
CUDA_VISIBLE_DEVICES=3 python model_main_tf2.py --model_dir=/workspace/tf_object_detection/training/experiments/models4 --pipeline_config_path=/workspace/tf_object_detection/training/experiments/models4/ssd_efficient_1024.config --checkpoint_dir=/workspace/tf_object_detection/training/experiments/models4
```
#### Terminal 3
```
tensorboard --bind_all --port 8888 --logdir=training
```

***Important:*** To stop the training just ```ctrl + c```.

### 5. Scripts for testing the trained model
To test the trained model, please execute the following commands:

```
CUDA_VISIBLE_DEVICES=6 python exporter_main_v2.py --input_type image_tensor --pipeline_config_path /workspace/tf_object_detection/training/experiments/models4/ssd_efficient_1024.config --trained_checkpoint_dir /workspace/tf_object_detection/training/experiments/models4 --output_directory /workspace/tf_object_detection/training/experiments/models4/experiment1
```

```
CUDA_VISIBLE_DEVICES=6 python inference_video.py --labelmap_path /workspace/tf_object_detection/label_maps/bstld_label_map.pbtxt --model_path /workspace/tf_object_detection/training/experiments/models4/experiment1/saved_model --tf_record_path /workspace/tf_object_detection/tfrecords/test/test.tfrecords --config_path /workspace/tf_object_detection/training/experiments/models4/ssd_efficient_1024.config --output_path animation.mp4
```

***Important:*** The duration of the video can be modified by modifying the line 96 (frames) ```inference_video.py```. For more details refer to [video_configuration](https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/) 

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt

