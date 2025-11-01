# SDZWA Southwest classifier v3 Deployment Instructions

The SDZWA Southwest v3 classifier was trained by Kyra Swanson at the San Diego Zoo Wildlife Association's [Conservation Tech Lab](https://github.com/conservationtechlab/).

It was on an efficientnet model architecture in PyTorch.

The following instructions are for deploying the PyTorch model to a Sagemaker Serverless Endpoint served in a Torchserve container. In order to create and deploy the model archive from scratch, we need to work across two different environments:

1. your local environment, where you will:

   1. Download the model weights and class list CSV from s3
   2. load the model weights into PyTorch and re-compile to torchscript for CPU
   3. Install dependencies for `torch-model-archiver` and run `torch-model-archiver` to generate the `.mar` file (a bundled archive that includes the torchscript-compiled model and the hander function)
   4. [Optionally] test the model and handler in a torchserve Docker container by building it and requesting inference locally
   5. Upload the model archive to s3

2. The Sagemaker notebook environment where you will:

   1. download the .mar archive from s3
   2. build the deploy image and push to ECR
   3. create a serverless endpoint configuration
   4. deploy and test a serverless endpoint

## Download weights model

From this directory, run:

```bash
aws s3 sync s3://animl-model-zoo/sdzwa-southwestv3/ model-weights
```

> **NOTE:** if there's a 'sdzwa-southwestv3_compiled_cpu.pt' file present, you can skip the next step and jump to creating the `.mar` file.

## Load the weights into PyTorch locally and re-compile to torchserve for CPU

Create and activate the Conda environment by running the following form this directory:

```bash
conda env create -f environment.yml
conda activate southwest-classifier
```

Then step through `sdzwa-southwestv3_compile.ipynb`. The notebook should produce a torchscript model 'sdzwa-southwestv3_compiled_cpu.pt' in the `./model-weights/` directory.

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
torch-model-archiver --model-name sdzwa-southwestv3 --version 3.0.0 --serialized-file model-weights/southwest_v3_compiled_cpu.pt --extra-files index_to_name.json --handler sdzwa-southwestv3_handler.py
mv sdzwa-southwestv3.mar model-store/sdzwa-southwestv3.mar
```

## Locally build, serve, and test the torchscript model with torchserve

We can now locally test this model prior to deploying.

Build the Docker image (you only have to do this once or if you've modified the Dockerfile):

```bash
docker build -t torchserve-sdzwa-southwestv3:latest-cpu .
```

Run it:

```bash
bash docker_sdzwa-southwestv3.sh $(pwd)/model-store
```

A couple of things need to happen to test the endpoint locally via cURL. To build the payload we need to download an image to test (preferably from Animl because we likely already have bounding boxes for it in the correct format), read the test image into a shell environment as a base64 string, then save the string to a bash variable. If the image came from Animl and has an object in it, you'll also want to look up the test object's corresponding bounding box in the Animl database and save that to a variable, and then compose the JSON payload with [jq](https://stedolan.github.io/jq/download/) and finally send that payload to our torchserve endpoint via cURL.

The steps look like this (on a Mac). Just be sure to modify the variables for the image path and bounding box you're testing.

1. Build payload

```bash
IMG_STRING=$(base64 -i ~/Downloads/Southwest/2021-06-30_SYFR0218_copy.JPG)
BBOX=[0.37166744470596313,0.5412790179252625,0.6243616342544556,0.7837801575660706]
PAYLOAD=$( jq -n \
            --arg image "$IMG_STRING" \
            --arg bbox "$BBOX" \
            '{image: $image, bbox: $bbox}' )

```

2. Invoke endpoint with payload:

```bash
curl -i http://127.0.0.1:8080/invocations -F body=$PAYLOAD
```

> **NOTE:** the model can also be queried at `http://127.0.0.1:8080/predictions/sdzwa-southwestv3`, but to test the endpoint that is queried during production (i.e. the sagemaker endpoint, which uses the configurations set in deployment/config.properties to adjust threads, worker count, and other container parameters), use `/invocations`.

The result should look something like:

```json
{
  "cougar": 0.9996892213821411,
  "dog": 0.00017720309551805258,
  "human": 2.2916185116628185e-5,
  "fox": 2.152714841940906e-5,
  "cat": 2.0666810087277554e-5
}
```

# Deploying the model to a Sagemaker Serverless Endpoint

Once you have run the model archiver step above, you're ready to upload that model to s3 so it can be deployed to a serverless inference endpoint!

Run the following to copy the model to the appropriate s3 bucket where pytorch and tensorflow models (for MIRAv1) are stored:

```bash
aws s3 cp model-store/sdzwa-southwestv3.mar s3://animl-model-zoo/sdzwa-southwestv3/
```

You'll also need to push the locally built docker image to the ECR repository. Since the images are large, it is fastest to do this from a sagemaker notebook instance in the sdzwa-southwestv3_deploy.ipynb.

Start up a Sagemaker Notebook instance and associate this repo with it to pull in the `sdzwa-southwestv3_deploy.ipynb` and supporting files with it. Step through that notebook to (re)build and push the Docker image to ECR, zip up our `.mar` file to prep it for deployment, create the model, endpoint config, and endpoint in Sagemaker, and finally test the endpoint.
