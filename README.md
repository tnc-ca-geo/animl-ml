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
to debug and test endpoints locally before deploying, this repo contains git 
submodule verisons of AWS's 
[Sagemaker TensorFlow Serving Container repo](https://github.com/aws/sagemaker-tensorflow-serving-container/) 
and [Microsoft's CameraTraps repo](https://github.com/microsoft/CameraTraps) 
and instructions below to help you run the container, load models, dependencies, 
and pre/postprocessing scripts into it, and submit requests to the local 
endpoints for inference.

## Getting started

## Local development and experimentation

NOTE: this assumes that you have 
[aws-vault](https://github.com/99designs/aws-vault) installed. 



