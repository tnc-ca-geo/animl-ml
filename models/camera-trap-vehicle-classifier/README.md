# Camera Trap Vehicle Classifier Deployment Instructions

The Camera Trap Vehicle Classifier was trained by Dan Morris and its full repository be found [here](https://github.com/agentmorris/camera-trap-vehicle-classifier).

It's a PyTorch model, fine-tuned from the [timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k](https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k) base model, that classifies vehicles cropped from camera trap images into the following categories:

- car/truck
- motorbike
- mountain bike
- quad

The following instructions are for deploying Torch model to a Sagemaker Serverless Endpoint served in a Torchserve container. In order to create and deploy the model archive from scratch, we need to work across two different environments:

1. your local environment, where you will:

   1. Download the model weights and class list from s3
   2. load the model weights into PyTorch and re-compile to torchscript for CPU
   3. Install dependencies for `torch-model-archiver` and run `torch-model-archiver` to generate the `.mar` file (a bundled archive that includes the torchscript-compiled model and the hander function)
   4. [Optionally] test the model and handler in a torchserve Docker container by building it and requesting inference locally
   5. Upload the model archive to s3

2. The Sagemaker notebook environment where you will:

   1. download the .mar archive from s3
   2. build the deploy image and push to ECR
   3. create a serverless endpoint configuration
   4. deploy and test a serverless endpoint

## Download and unzip model checkpoint and classes

From this directory, run:

```bash
aws s3 sync s3://animl-model-zoo/camera-trap-vehicle-classifier/ .
unzip original-model/camera-trap-vehicle-classifier.2025.07.09.zip -d original-model/
rm original-model/camera-trap-vehicle-classifier.2025.07.09.zip
```

You should have a directory structure that looks like:

```
...
/irc
    |-- exported-model
        |-- index_to_name.json
        |-- // TODO: UPDATE WITH EXPORTED FILE NAME
    |-- original-model
        |-- classes.txt
        |-- camera-trap-vehicle-classifier.2025.07.09.ckpt
    ...
```

> **NOTE:** if there's also a <TODO: UPDATE WITH FILE NAME> file present, that's the older/current Torchscript Model Archive, and unless you want to re-compile this model for CPU and create a new `.mar` file(perhaps because the weights or the inference code changed), you can skip to <TODO: UPDATE WITH STEP>

## Load the weights into PyTorch locally and re-compile to torchserve for CPU

Create and activate a Conda environment and install dependencies by running the following form this directory:

```bash
conda create -n camera-trap-vehicle-classifier python=3.11 pip -y
conda activate camera-trap-vehicle-classifier
pip install -r requirements.txt
```
