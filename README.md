# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* To Predict Object Detection using Tensorflow Object Detection API on Raspberry PI
* Version 1.0

### How do I get set up? ###
#### Install Tensorflow and other dependencies: ####

	sudo apt-get install libblas-dev liblapack-dev python-dev libatlas-base-dev gfortran python-setuptools
	sudo pip install http://ci.tensorflow.org/view/Nightly/job/nightly-pi-zero/lastSuccessfulBuild/artifact/output-artifacts/tensorflow-1.4.0-cp27-none-any.whl


#### Clone Tensorflow Object Detection API: ####

git clone https://github.com/tensorflow/models.git

#### Configure API using  protobuf ####
	sudo apt-get install -y protobuf-compiler
	cd models/research/
	protoc object_detection/protos/*.proto --python_out=.

##### add path to bashrc #####
	export PYTHONPATH=$PYTHONPATH:/home/pi/models/research:/home/pi/models/research/slim

#### Download Pre-Trained Models ####
all different pre-trained models are available at [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
	
	wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
	tar -zxvf ssd_mobilenet_v1_coco_11_06_2017.tar.gz

##### other Packages required #####
	sudo apt-get install libjpeg-dev
	sudo pip install Pillow

#### Predict ####
clone this repo and run
	
	python ObjectDetectionPredict.py
