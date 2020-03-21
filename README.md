# Animl ML
Machine Learning resources for camera trap data processing

## Intro

We are using AWS Sagemaker to host our model endpoints. The initial models
we will run inference on are 
[Microsoft's Megadetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md),
an model to detect animals, people, and vehicles in camera trap images, and 
[MIRA](https://github.com/tnc-ca-geo/mira), a species classifier trained on 
labeled images from Santa Cruz Island.

When you deploy model endpoints to Sagemaker, AWS starts an EC2 instance and 
starts a docker container with in it optimized for serving particular model's 
architecture. You can find Phython Nodetbooks to facilitate the deployment of 
models in the notebooks/ directory of this repo.

Additionally, if you want to launch a TensorFlow serving container locally 
to debug and test endpoints locally before deploying, this repo contains a 
script to clone AWS's 
[Sagemaker TensorFlow Serving Container repo](https://github.com/aws/sagemaker-tensorflow-serving-container/) 
and [Microsoft's CameraTraps repo](https://github.com/microsoft/CameraTraps) 
and instructions below to help you run the container, load models, dependencies, 
and pre/postprocessing scripts into it, and submit requests to the local 
endpoints for inference.

## Get the models


## Local development and experimentation

NOTE: this assumes that you have 
[aws-vault](https://github.com/99designs/aws-vault) and 
[docker](https://docs.docker.com/docker-for-mac/install/)installed. 

### Get the CameraTrap and Sagemaker contianer repos
After cloning this repo, cd into the ```animl-ml/``` project directory, and 
run the script to clone the necessary external repos:

```
$ cd animl-ml
$ bash ./scripts/get-libs.sh
```

### Building the container
To build the docker container, cd into the sagemaker repo and run the build script:
```
$ cd ../sagemaker-tensorflow-serving-container/
$ aws-vault exec `_var_`profile`_var_` -- ./scripts/build.sh --version `_var_`tf version`_var_`1.12 --arch `_var_`architecture`_var_`
```

For example, to build a container with TensorFlow 1.12 on a CPU architecture 
(note - you can't do gpu on mac so you're limited to cpu), the comand would look like:
```
$ aws-vault exec home -- ../sagemaker-tensorflow-serving-container/scripts/build.sh --version 1.12 --arch cpu
```

Note - you may need to first export the AWS_DEFAULT_REGION variable manually 
before running the build script. 

```
$ export AWS_DEFAULT_REGION=us-west-1
```

You can double check that the container was built with 
```
$ docker images
```

### Running the container
To run the container, cd back to the ```animl-ml``` project directory and run 
the ```start-container.sh``` script

```
$ cd ../animl-ml
$ aws-vault exec `_var_`profile`_var_` -- ./scripts/start-container.sh --version `_var_`tf version`_var_`1.12 --arch `_var_`architecture`_var_`
```

To stop the container, run:
```
$ aws-vault exec `_var_`profile`_var_` -- ./scripts/stop-container.sh --version `_var_`tf version`_var_`1.12 --arch `_var_`architecture`_var_`
```