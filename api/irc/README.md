# Irvine Ranch Conservancy classifier Deployment Instructions

The IRC v2 classifier was trained by The Irvine Ranch Conservancy using the mewc-train v1 workflow, which trains an efficientnetv2 model architecture in Tensorflow v2.9.1 The model predicts 16 classes.

The following instructions are for converting the .h5 Keras model to a Tensorflow Saved Model, building a Docker container to serve the model via Tensorflow Serving, deploying the container to AWS ECR, and then deploying the container as an AWS Sagemaker Serverless Inference endpoint.
