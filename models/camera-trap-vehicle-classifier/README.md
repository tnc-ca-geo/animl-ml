# Camera Trap Vehicle Classifier Deployment Instructions

The Camera Trap Vehicle Classifier was trained by Dan Morris and its full repository be found [here](https://github.com/agentmorris/camera-trap-vehicle-classifier).

It's a PyTorch model, fine-tuned from the [timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k](https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k) (EVA-02) base model, that classifies vehicles cropped from camera trap images into the following categories:

- car/truck
- motorbike
- mountain bike
- quad

It expects an input of `(448, 448)`.

The following instructions are for deploying this PyTorch model to a Sagemaker Serverless Endpoint served in a Torchserve container. In order to create and deploy the model archive from scratch, we need to work across two different environments:

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
/camera-trap-vehicle-classifier
    |-- exported-model
        |-- camera-trap-vehicle-classifier_compiled_cpu.pt
        |-- camera-trap-vehicle-classifier.mar
        |-- index_to_name.json
    |-- original-model
        |-- classes.txt
        |-- camera-trap-vehicle-classifier.2025.07.09.ckpt
    ...
```

> **NOTE:** if there's a `.mar` file present in the `/exported-model` directory, that's the older/current Torchscript Model Archive, and unless you want to re-compile this model for CPU and create a new `.mar` file (perhaps because the weights or the inference code changed), you can skip to the [Locally build/test step](#locally-build-serve-and-test-the-torchscript-model-with-torchserve).

## Load the weights into PyTorch locally and re-compile to Torchserve for CPU

Create and activate a Conda environment and install dependencies by running the following form this directory:

```bash
conda create -n camera-trap-vehicle-classifier python=3.11 pip -y
conda activate camera-trap-vehicle-classifier
pip install -r requirements.txt
```

Then step through `camera-trap-vehicle-classifier_compile.ipynb`. The notebook should produce a torchscript model 'camera-trap-vehicle-classifier_compiled_cpu.pt' in the `./exported-model/` directory.

Note for others using these steps to deploy a different model: the versions of `torch` and `torchvision` that you pin in your `Dockerfile` used for serving must match the versions you use when compiling the model to torchscript. To check which versions you're using in your venv use `pip freeze` and to bump the versions up (or down) use `pip install --upgrade` (e.g. `pip install --upgrade torchvision==0.15.1`).

## Install and run `torch-model-archiver` to generate .mar file

Full documentation for creating a torchserve model archive (.mar) file can be found [here](https://github.com/pytorch/serve/tree/master/model-archiver#creating-a-model-archive).

> **NOTE:** because we want to crop images to their respective bounding boxes and resize them to match the resizing and transformations that were performed during training, we created a [custom handler](https://github.com/pytorch/serve/blob/master/docs/custom_service.md#custom-handlers). However, if you are trying to follow these steps to deploy a different image classifier and don't need to do any pre-processing, passing in one of the [default handlers](https://github.com/pytorch/serve/blob/master/docs/default_handlers.md) (i.e. ` --handler image_classifier`) to the `torch-model-archiver` works fine as an alternative.

Run:

```bash
pip install torch-model-archiver
```

to install dependencies, then the following to create the archive:

```bash
torch-model-archiver --model-name camera-trap-vehicle-classifier --version 1.0.0 --serialized-file exported-model/camera-trap-vehicle-classifier_compiled_cpu.pt --extra-files exported-model/index_to_name.json --handler camera-trap-vehicle-classifier_handler.py
mv camera-trap-vehicle-classifier.mar exported-model/camera-trap-vehicle-classifier.mar
```

Finally, upload the exported files to S3 so that they are accessible for deployment and future use:

```bash
aws s3 cp --recursive ./exported-model s3://animl-model-zoo/camera-trap-vehicle-classifier/exported-model
```

## Locally build, serve, and test the Torchscript model with Torchserve

We can now locally test this model prior to deploying.

Build the Docker image (you only have to do this once or if you've modified the Dockerfile):

```bash
docker build -t camera-trap-vehicle-classifier:latest-cpu .
```

Run it:

```bash
bash docker-run.sh $(pwd)/exported-model
```

Test it:

```bash
# compose the JSON payload with jq (https://stedolan.github.io/jq/download/)
# and send that payload to our torchserve endpoint via cURL.
# To test other images locally, modify the variables for the image path
# and bounding box accordingly. BBOXes are [ymin, xmin, ymax, xmax], normalized (not absolute)

# build payload (mountain biker test)
IMG_STRING=$(base64 -i ./tests/test-data/mountain-biker-test.jpg) \
BBOX=[0.19789853692054749,0,0.9022471904754639,0.5614370703697205] \
PAYLOAD=$( jq -n \
            --arg image "$IMG_STRING" \
            --arg bbox "$BBOX" \
            '{image: $image, bbox: $bbox}' )


# invoke endpoint with payload:
curl -i http://127.0.0.1:8080/invocations -F body=$PAYLOAD
```

The result should look something like:

```json
{
  "mountain_biker": 0.9999924898147583,
  "motorbike": 5.223560037848074e-6,
  "car_or_truck": 1.3966878213977907e-6,
  "quad": 9.004459684547328e-7
}
```

## Deploying the model to a Sagemaker Serverless Endpoint

Start up a Sagemaker Notebook instance and associate this repo with it to pull in the `camera-trap-vehicle-classifier_deploy.ipynb` and supporting files with it. Step through that notebook to (re)build and push the Docker image to ECR, create the model, endpoint config, and endpoint in Sagemaker, and finally test the endpoint.
