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

## Local development and experimentation

NOTE: this assumes that you have 
[aws-vault](https://github.com/99designs/aws-vault),  
[docker](https://docs.docker.com/docker-for-mac/install/) and 
[virtualenv](https://virtualenv.pypa.io/en/latest/) installed.

### Clone the repo and set up the virtual env
```
$ mkdir animl-ml
$ git clone https://github.com/tnc-ca-geo/animl-ml.git
$ virtualenv env -p python3
$ pip install -r requirements.txt
```

### Get the CameraTrap and Sagemaker contianer repos
After cloning this repo, cd into the ```animl-ml/``` project directory, and 
run the script to clone the necessary external repos:

```
$ cd animl-ml
$ bash ./scripts/get-libs.sh
```

## Get the models
To download and unzip the models, run the following from 
the same ```animl-ml``` directory:
```
bash scripts/get-models.sh
```

### Building the container
To build the docker container, cd into the sagemaker repo and run the build script:
```
$ cd ../sagemaker-tensorflow-serving-container/
$ aws-vault exec <vault_profile> -- ./scripts/build.sh --version <tf_version> --arch <architecture_type>
```

For example, to build a container with TensorFlow 1.12 on a CPU architecture 
(note - you can't run gpu architecture on mac), the comand would look like:
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
$ aws-vault exec <vault_profile> -- ./scripts/start-container.sh --version <tf_version> --arch <architecture_type>
```

Alternatively, you can also start the container in interactive mode with:
```
$ aws-vault exec <vault_profile> -- ./scripts/start-container-interactive.sh --version <tf_version> --arch <architecture_type>
```

To stop the container, run:
```
$ aws-vault exec <vault_profile> -- ./scripts/stop-container.sh --version <tf_version> --arch <architecture_type>
```

### Run inference on the local endpoint
To test the endpoint, pass the ```make-request.py``` script a string representing 
the key of an image that's already been uploaded to the ```animl-images``` S3 bucket. 
The preprocessing script in ```inference.py``` will download the image, process 
it, and submit it to the model for inference.

```
$ aws-vault exec <vault_profile> -- python scripts/make-request.py  <image_key>
```