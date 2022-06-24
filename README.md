# Animl ML
Machine Learning resources for camera trap data processing

## `Related repos`
- Animl Fontend           http://github.com/tnc-ca-geo/animl-frontend
- Animl API               http://github.com/tnc-ca-geo/animl-api
- Animl base program      http://github.com/tnc-ca-geo/animl-base
- Animl ingest function   http://github.com/tnc-ca-geo/animl-ingest
- Animl desktop app       https://github.com/tnc-ca-geo/animl-desktop

## `Intro`

We are using AWS Sagemaker to host our model endpoints. The initial models we will run inference on are [Microsoft's Megadetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md), a model to detect animals, people, and vehicles in camera trap images, and [MIRA](https://github.com/tnc-ca-geo/mira), a pair of species classifiers trained on labeled images from Santa Cruz Island.

This repo contains: 
  - Python Notebooks to facilitate loading and deploying models as endpoints on AWS Sagemaker
  - Resources for running & debugging model endpoints locally
  - A Serverless API for submitting images & bounding boxes to the MIRA endpoints for real-time inference
  - A notebook with code examples for invoking the APIs and testing the inference pipeline end-to-end

## `Test the inference pipeline`

The most fun place to start is the ```notebooks/test-inference-pipeline.ipynb```. Fire it up and step though the notebook to test submitting images to the Megadetector API for object detection and then to MIRA API for species classification.

NOTE: you will need a Megadetector API key in order to use their API. 

## `Deploy a model endpoint using AWS Sagemaker Notebook`

When you deploy model endpoints to Sagemaker, AWS starts an EC2 instance and starts a docker container with in it optimized for serving particular model's architecture. You can find Python Notebooks to facilitate the deployment of models in the ```notebooks/``` directory of this repo.

To deploy a model endpoint, start up a Sagemaker notebook instance in AWS, associate this repo with it, and step through one of the deployment notebooks in the ```notebooks/``` directory to get started.

## `Local endpoint development and debugging`

If you want to launch a TensorFlow serving container locally to debug and test endpoints on your computer before deploying, this repo contains a script to clone AWS's [Sagemaker TensorFlow Serving Container repo](https://github.com/aws/sagemaker-tensorflow-serving-container/) and [Microsoft's CameraTraps repo](https://github.com/microsoft/CameraTraps) and instructions below to help you run the container, load models, dependencies, and pre/postprocessing scripts into it, and submit requests to the local endpoints for inference.

NOTE: be sure that you have the following installed:
 - [aws-vault](https://github.com/99designs/aws-vault)
 - [docker](https://docs.docker.com/docker-for-mac/install/) 
 - [virtualenv](https://virtualenv.pypa.io/en/latest/) 

### 1. Clone the repo and set up the virtual env

```
$ mkdir animl-ml
$ git clone https://github.com/tnc-ca-geo/animl-ml.git
$ virtualenv env -p python3
$ source env/bin/activate
$ cd animl-ml
$ pip3 install -r requirements.txt
```

### 2. Get the CameraTrap and Sagemaker contianer repos

After cloning this repo, from the ```animl-ml/animl-ml/``` project directory, run the script to clone the necessary external repos:
```
$ bash ./scripts/get-libs.sh
```

### 3. Get the models

The models we use in this app are all available in Tensorflow ProtoBuf format at s3://animl-model-zoo. To download and unzip them, run the following from the same ```animl-ml``` project directory:
```
$ aws-vault exec <vault_profile> -- bash ./scripts/get-models.sh
```

NOTE: If you're on a mac, make sure there aren't any stray ```.DS_store``` files in ```animl-ml/models/```. The sagemaker-tensorflow-serving-container build scripts will mistake them for models and try to load them into the container. A quick way to recursively remove all ```.DS_store``` files is to cd to the ```animl-ml/models/``` directory and run: 
```
$ find . -name '*.DS_Store' -type f -delete
```

### 4. Building the container

And finally, to build the docker container in which the model will be run locally, execute:
```
$ aws-vault exec <vault_profile> -- bash ./scripts/build-container.sh
```

### 5. Running the container

To run the container, run the ```start-container.sh``` script
```
$ aws-vault exec <vault_profile> -- bash ./scripts/start-container.sh
```

Check that it was successful and the container is running with:
```
$ docker ps
``` 

Alternatively, you can also start the container in interactive mode with:
```
$ aws-vault exec <vault_profile> -- bash ./scripts/start-container-interactive.sh
```

To stop the container, run:
```
$ aws-vault exec <vault_profile> -- bash ./scripts/stop-container.sh
```

All output from the container will be piped into ```log.txt```.

### 6. Run inference on the local endpoint
To test the endpoint, pass the ```make-request.py``` script a path to an local image file:
```
$ aws-vault exec <vault_profile> -- python ./scripts/make-request.py input/sample-img.jpg
```
