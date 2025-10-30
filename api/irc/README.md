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

## Deploying the model to a Sagemaker Serverless Endpoint

Use a SageMaker Notebook instance to run the `deploy_to_sagemaker.ipynb` notebook. The notebook walks through creating model on SageMaker, preparing the endpoint, deploying, and testing.
