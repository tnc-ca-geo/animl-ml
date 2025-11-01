# MIRAv2 Deployment Instructions

MIRAv2 was trained on an efficientnet model architecture in PyTorch following the workflow described in the `/classification/` directory of this repo (which itself was derived from Microsoft's AI for Earth team's [MegaClassifier training](https://github.com/microsoft/CameraTraps/tree/main/classification) workflow).

The following instructions are for deploying the PyTorch model to a Sagemaker Serverless Endpoint served in a Torchserve container. In order to create and deploy the model archive from scratch, we need to work across two different environments:

1. your local environment, where you will:
   1. Download the model weights from s3 
   2. load the model weights into PyTorch and re-compile to torchscript for CPU (TODO: double check that this is necessary)
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
aws s3 sync s3://animl-model-zoo/mirav2/ model-weights
```
> **NOTE:** 'ckpt_18_compiled.pt' is a compiled torchscript model, but it was compiled on GPU so unlikely to work on Sagemaker Serverless endpoints, which are CPU only. For now we only really care about the un-compiled weights in 'ckpt_18.pt'.

Also, if there's a 'mira_compiled_cpu.pt' file present, you can skip the next step and jump to creating the `.mar` file.

## Load the weights into PyTorch locally and re-compile to torchserve for CPU

Start the `venv` located one level above the root directory of this project (or create one if one doesn't exist). Next, install the [PyTorch Efficientnet implementaion](https://github.com/lukemelas/EfficientNet-PyTorch) that was used during training with: 
```bash
pip install efficientnet_pytorch
```

Then step through `mirav2_compile.ipynb`. The notebook should produce a torchscript model 'mira_compiled_cpu.pt' in the `./model-weights/` directory.

Note for others using these steps to deploy a different model: the versions of `torch` and `torchvision` that you pin in your `Dockerfile` used for serving must match the versions you use when compiling the model to torchscript. To check which versions you're using in your venv use `pip freeze` and to bump the versions up (or down) use `pip install --upgrade` (e.g. `pip install --upgrade torchvision==0.11.1`).


## Install and run `torch-model-archiver` to generate .mar file

Full documentation for creating a torchserve model archive (.mar) file can be found [here](https://github.com/pytorch/serve/tree/master/model-archiver#creating-a-model-archive).

> **NOTE:** because we want to crop images to their respective bounding boxes and resize them to match the resizing and transformations that were performed during training, we created a [custom handler](https://github.com/pytorch/serve/blob/master/docs/custom_service.md#custom-handlers). However, if you are trying to follow these steps to deploy a different image classifier and don't need to do any pre-processing, passing in one of the [default handlers](https://github.com/pytorch/serve/blob/master/docs/default_handlers.md) (i.e. ` --handler image_classifier`) to the `torch-model-archiver` works fine as an alternative.

Run:
```bash
pip install torch-model-archiver
```

to install dependencies, then the following to create the archive:
```bash
torch-model-archiver --model-name mira --version 2.0.0 --serialized-file model-weights/mira_compiled_cpu.pt --extra-files index_to_name.json --handler mirav2_handler.py
mv mira.mar model_store/mira.mar
```

## Locally build, serve, and test the torchscript model with torchserve
We can now locally test this model prior to deploying.

Build the Docker image (you only have to do this once or if you've modified the Dockerfile):
```bash
docker build -t torchserve-mirav2:0.5.3-cpu .
```

Run it:
```bash
bash docker_mirav2.sh $(pwd)/model_store
```

A couple of things need to happen to test the endpoint locally via cURL. To build the payload we need to download an image to test (preferably from Animl because we likely already have bounding boxes for it in the correct format), read the test image into a shell environment as a base64 string, then save the string to a bash variable. If the image came from Animl and has an object in it, you'll also want to look up the test object's corresponding bounding box in the Animl database and save that to a variable, and then compose the JSON payload with [jq](https://stedolan.github.io/jq/download/) and finally send that payload to our torchserve endpoint via cURL. 

The steps look like this (on a Mac). Just be sure to modify the variables for the image path and bounding box you're testing. 

1. Build payload
```bash
IMG_STRING=$(base64 -i ~/Downloads/rodent-full-size.jpg)
BBOX=[0.3312353789806366,0.22853891551494598,0.4267701208591461,0.436643123626709]
PAYLOAD=$( jq -n \
            --arg image "$IMG_STRING" \
            --arg bbox "$BBOX" \
            '{image: $image, bbox: $bbox}' )

```

2. Invoke endpoint with payload:
```bash
curl -i http://127.0.0.1:8080/invocations -F body=$PAYLOAD
```

> **NOTE:** the model can also be queried at `http://127.0.0.1:8080/predictions/mira`, but to test the endpoint that is queried during production (i.e. the sagemaker endpoint, which uses the configurations set in deployment/config.properties to adjust threads, worker count, and other container parameters), use `/invocations`.

The result should look something like:

```json
{
  "rodent": 1.0,
  "bird": 2.4474083870629215e-10,
  "fox": 1.882403434516622e-11,
  "lizard": 1.728995766003827e-11,
  "skunk": 8.361315654605017e-14
}
```


# Deploying the model to a Sagemaker Serverless Endpoint

Once you have run the model archiver step above, you're ready to upload that model to s3 so it can be deployed to a serverless inference endpoint!

Run the following to copy the model to the appropriate s3 bucket where pytorch and tensorflow models (for MIRAv1) are stored:
```bash
aws s3 cp model_store/mira.mar s3://animl-model-zoo/
```

You'll also need to push the locally built docker image to the ECR repository. Since the images are large, it is fastest to do this from a sagemaker notebook instance in the mirav2_deploy.ipynb.

Start up a Sagemaker Notebook instance and associate this repo with it to pull in the `mirav2_deploy.ipynb` and supporting files with it. Step through that notebook to (re)build and push the Docker image to ECR, zip up our `.mar` file to prep it for deployment, create the model, endpoint config, and endpoint in Sagemaker, and finally test the endpoint.