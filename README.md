# README #

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

#### Copy trained model on pi ####
Train a model to run on raspberry pi using this repo

https://github.com/NanoNets/RaspberryPi-ObjectDetection-TensorFlow

Copy exported model and label file from data directory to raspberry pi 

##### other Packages required #####
	sudo apt-get install libjpeg-dev
	sudo pip install Pillow

#### Predict ####
clone this repo and run
	
	python ObjectDetectionPredict.py --model data/0/quantized_graph.pb --labels data/label_map.pbtxt --images /data/image1.jpg /data/image2.jpg
