# Irvine Ranch Conservancy classifier Deployment Instructions

The IRCv2 classifier was trained by The Irvine Ranch Conservancy using the [mewc-train](https://github.com/zaandahl/mewc-train) v1 workflow, which trains an efficientnetv2 model architecture in Tensorflow v2.9.1 The model predicts 16 classes, and expects an input of 300x300px.

The following instructions are for converting the original .h5 Keras model to a Tensorflow Saved Model, building a Docker container to serve the model via Tensorflow Serving wrapped in a FastAPI application, deploying the container to AWS ECR, and then deploying the container as an AWS Sagemaker Serverless Inference endpoint.

This workflow could be adapted for any Tensorflow or Keras model in the future.

In order to create and deploy the IRCv2 model from scratch, we need to work across two different environments:

1. your local environment, where you will:

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

## Deploying the model to a Sagemaker Serverless Endpoint

Once you have completed the steps above, you're ready to upload that model to s3 so it can be deployed to a serverless inference endpoint!

Run the following to copy the model to the appropriate s3 bucket where pytorch and tensorflow models (for MIRAv1) are stored:

```bash
aws s3 cp ./tf-saved-model.tar.gz s3://animl-model-zoo/irc/
```

Use a SageMaker Notebook instance to run the `deploy_to_sagemaker.ipynb` notebook. The notebook walks through creating model on SageMaker, preparing the endpoint, deploying, and testing.
