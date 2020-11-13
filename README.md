# Animl ML
Machine Learning resources for camera trap data processing

## `Related repos`
- Animl Fontend           http://github.com/tnc-ca-geo/animl-frontend
- Animl API               http://github.com/tnc-ca-geo/animl-api
- Animl base program      http://github.com/tnc-ca-geo/animl-base
- Animl ingest function   http://github.com/tnc-ca-geo/animl-ingest
- Animl desktop app       https://github.com/tnc-ca-geo/animl-desktop

## `Intro`

We are using AWS Sagemaker to host our model endpoints. The initial models
we will run inference on are 
[Microsoft's Megadetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md),
an model to detect animals, people, and vehicles in camera trap images, and 
[MIRA](https://github.com/tnc-ca-geo/mira), a species classifier trained on 
labeled images from Santa Cruz Island.

When you deploy model endpoints to Sagemaker, AWS starts an EC2 instance and 
starts a docker container with in it optimized for serving particular model's 
architecture. You can find Phython Nodetbooks to facilitate the deployment of 
models in the ```notebooks/``` directory of this repo.

Additionally, if you want to launch a TensorFlow serving container locally 
to debug and test endpoints locally before deploying, this repo contains a 
script to clone AWS's 
[Sagemaker TensorFlow Serving Container repo](https://github.com/aws/sagemaker-tensorflow-serving-container/) 
and [Microsoft's CameraTraps repo](https://github.com/microsoft/CameraTraps) 
and instructions below to help you run the container, load models, dependencies, 
and pre/postprocessing scripts into it, and submit requests to the local 
endpoints for inference.

## `Deploy a model endpoint using AWS Sagemaker Notebook`
The ```notebooks/``` directory contains notebooks that can be pulled into an 
AWS Sagemaker notebook instance and used as is or repurposed for deploying 
endpoints into production and testing them. We currently have a notebook 
instance running, which can be found here: 

https://animl.notebook.us-west-1.sagemaker.aws/lab

## `Local development and experimentation`

NOTE: this assumes that you have 
[aws-vault](https://github.com/99designs/aws-vault),  
[docker](https://docs.docker.com/docker-for-mac/install/) and 
[virtualenv](https://virtualenv.pypa.io/en/latest/) installed.

### Clone the repo and set up the virtual env
```
$ mkdir animl-ml
$ git clone https://github.com/tnc-ca-geo/animl-ml.git
$ virtualenv env -p python3
$ source env/bin/activate
$ cd animl-ml
$ pip install -r requirements.txt
```

### Get the CameraTrap and Sagemaker contianer repos
After cloning this repo, from the ```animl-ml/``` project directory,  
run the script to clone the necessary external repos:
```
$ bash ./scripts/get-libs.sh
```

### Get the models
To download and unzip the models, run the following from 
the same ```animl-ml``` directory:
```
$ bash ./scripts/get-models.sh
```
NOTE: If you're on a mac, make sure there aren't any stray ```.DS_store``` 
files in ```animl-ml/models/```. The sagemaker-tensorflow-serving-container 
build scripts will mistake them for models and try to load them into the 
container. A quick way to recursively remove all ```.DS_store``` files is to 
cd to the ```animl-ml/models/``` directory and run: 
```
$ find . -name '*.DS_Store' -type f -delete
```

### Building the container
And finally, to build the docker container in which the model will be run 
locally, execute:
```
$ aws-vault exec <vault_profile> -- bash ./scripts/build-container.sh
```

### Running the container
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

### Run inference on the local endpoint
To test the endpoint, pass the ```make-request.py``` script a path to an local 
image file:
```
$ aws-vault exec <vault_profile> -- python ./scripts/make-request.py input/sample-img.jpg
```