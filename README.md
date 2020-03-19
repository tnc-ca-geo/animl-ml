# Animl ML
Machine Learning resources for camera trap data processing

## Intro

We are using AWS Sagemaker to host our model endpoints. The initial models
we will run inference on are 
[Microsoft's Megadetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md),
an model to detect animals, people, and vehicles in camera trap images, and 
[MIRA](https://github.com/tnc-ca-geo/mira), a species classifier trained on 
labeled images from Santa Cruz Island and.

When you deploy model endpoints to Sagemaker, AWS starts an EC2 instance and 
starts a docker containers optimized for serving particular model's 
architecture. You can find Phython Nodetbooks to facilitate the deployment of 
models in the notebooks/ directory of this repo.

Additionally, if you want to debug containers and test endpoints locally before 
deploying, this repo contains git submodule verisons of 
[Sagemaker TensorFlow Serving Container repo](https://github.com/aws/sagemaker-tensorflow-serving-container/) 
and [Microsoft's CameraTraps repo](https://github.com/microsoft/CameraTraps) 
and instructions to help you launch a sagemaker docker container on your 
comptuer and experiement with running inference on models before you deploy.



