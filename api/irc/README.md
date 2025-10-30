# Irvine Ranch Conservancy classifier Deployment Instructions

The IRC v2 classifier was trained by The Irvine Ranch Conservancy using the mewc-train v1 workflow, which trains an efficientnetv2 model architecture in Tensorflow v2.9.1 The model predicts 16 classes.

The following instructions are for converting the original .h5 Keras model to a Tensorflow Saved Model, building a Docker container to serve the model via Tensorflow Serving wrapped in a FastAPI application, deploying the container to AWS ECR, and then deploying the container as an AWS Sagemaker Serverless Inference endpoint.

This workflow could be adapted for any Tensorflow or Keras model in the future.

## Deploying the model to a Sagemaker Serverless Endpoint

Once you have completed the steps above, you're ready to upload that model to s3 so it can be deployed to a serverless inference endpoint!

Run the following to copy the model to the appropriate s3 bucket where pytorch and tensorflow models (for MIRAv1) are stored:

```bash
aws s3 cp ./tf-saved-model.tar.gz s3://animl-model-zoo/irc/
```

Use a SageMaker Notebook instance to run the `deploy_to_sagemaker.ipynb` notebook. The notebook walks through creating model on SageMaker, preparing the endpoint, deploying, and testing.
