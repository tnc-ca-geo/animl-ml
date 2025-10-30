# Irvine Ranch Conservancy classifier Deployment Instructions

The IRCv2 classifier was trained by The Irvine Ranch Conservancy using the [mewc-train](https://github.com/zaandahl/mewc-train) v1 workflow, which trains an efficientnetv2 model architecture in Tensorflow v2.9.1 The model predicts 16 classes, and expects an input of 300x300.

The following instructions are for converting the original .h5 Keras model to a Tensorflow Saved Model, building a Docker container to serve the model via Tensorflow Serving wrapped in a FastAPI application, deploying the container to AWS ECR, and then deploying the container as an AWS Sagemaker Serverless Inference endpoint.

More specifically, in order to create and deploy the IRCv2 model from scratch, we need to work across two different environments:

1. Your local environment, where you will:

   1. Download the model weights from s3
   2. Convert the Keras model to Tensorflow Saved Model format
   3. Convert the original class_list.yaml to json
   4. Zip the Saved Model bundle and upload it to S3
   5. [Optionally] build, run, and invoke the container locally for testing

2. The Sagemaker notebook environment where you will:

   1. download the .tar Saved Model bundle
   2. build the deploy image and push to ECR
   3. create a serverless endpoint configuration
   4. deploy and test a serverless endpoint

This workflow could be adapted for any Tensorflow or Keras model in the future.

## Convert Keras weights to Tensorflow Saved Model

From this directory, run:

```bash
aws s3 sync s3://animl-model-zoo/irc/ .
```

You should have a directory structure that looks like:

```
...
/irc
    |-- exported-model
        |-- index_to_name.json
        |-- tf-saved-model.tar.gz
    |-- original-model
        |-- class_list.yaml
        |-- IRC_2.h5
        |-- variable.json
    ...
```

> **NOTE:** if there's also a `exported-model/tf-saved-model.tar.gz` file present, that's the older/current tarred Tensorflow Saved Model, and unless you want to re-convert the Keras model into a Saved Model bundle (perhaps because the weights changed), you can skip the rest of this step.

Create and activate the Conda environment by running the following form this directory:

```bash
conda env create -f environment.yml
conda activate southwest-classifier
```

Then step through `convert-keras-to-savedmodel.ipynb`. The notebook should produce a Tensorflow Saved Model bundle in the `tf-saved-model/` directory and create a tar file of that directory called `tf-saved-model.tar.gz`. If one already exists, replace old tf-saved-model.tar.gz with the new one:

```bash
mv tf-saved-model.tar.gz ./exported-model/tf-saved-model.tar.gz
```

Next, step through the `format-class-list.ipynb` notebook to convert the `original-model/class_list.yaml` to a JSON document. NOTE: this step is specific to this model (or perhaps to models trained with the mewc-train workflow), so you may need to adjust the class formatting notebook if you are using this to deploy a different Keras/Tensorflow model.

Finally, upload the exported files to S3 so that they are accessible for deployment and future use:

```bash
aws s3 cp --recursive ./exported-model s3://animl-model-zoo/irc/exported-model
```

## Running the FastAPI/Tensorflow Serving container locally

The FastAPI Server (serving/serve.py) provides support to deploy our custom Tensorflow Serving container on SageMaker. It exposes the necessary /ping and /invocation API routes for SageMaker hosting and provides some image pre-processing steps before passing the request on to Tensorflow Serving for a prediction.

To run it locally make sure you have the latest [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed. If you are running Docker Desktop on Apple Silicon chips (M1/M2/M3 etc.), it's critical that you choose "Docker VMM" as your Virtual Machine Manager in your Docker Desktop settings, as attempting to run amd64 images (which tensorflow-serving is) on an arm64 host can be problematic without it.

```bash
# build the container
docker build --platform=linux/amd64 -t irc-tf-fastapi .

# and run it
docker run --rm --platform=linux/amd64 -p 8080:8080 irc-tf-fastapi
```

### Testing the endpoint locally

A couple of things need to happen to test the endpoint locally via cURL. To build the payload we need an image to test (preferably from Animl because we likely already have bounding boxes for it in the correct format), read the test image into a shell environment as a base64 string, then save the string to a bash variable. If the image came from Animl and has an object in it, you'll also want to look up the test object's corresponding bounding box in the Animl database and save that to a variable, and then compose the JSON payload with [jq](https://stedolan.github.io/jq/download/) and finally send that payload to our torchserve endpoint via cURL.

The steps look like this (on a Mac). Just be sure to modify the variables for the image path and bounding box you're testing.

1. Build payload

```bash
IMG_STRING=$(base64 -i ./tests/test-data/coyote-test.jpg) \
BBOX=[0.4319087266921997,0.21275195479393005,0.6099987030029297,0.3272196650505066] \
PAYLOAD=$( jq -n \
            --arg image "$IMG_STRING" \
            --arg bbox "$BBOX" \
            '{image: $image, bbox: $bbox}' )

```

2. Invoke endpoint with payload:

```bash
curl -X POST http://127.0.0.1:8080/invocations \
  -H "Content-Type: application/json" \
  -d $PAYLOAD
```

The result should look something like:

```json
{
  "coyote": 0.990233779,
  "mountain lion": 0.0020499716,
  "mule deer": 0.00128549989,
  "bobcat": 0.00125635625,
  "gray fox": 0.000983441598
}
```

## Deploying the model to a Sagemaker Serverless Endpoint

Start up a Sagemaker Notebook instance and associate this repo with it to pull in the `deploy_to_sagemaker.ipynb` and supporting files with it. Step through that notebook to (re)build and push the Docker image to ECR, create the model, endpoint config, and endpoint in Sagemaker, and finally test the endpoint.
