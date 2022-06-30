This section for deploying Megadetector V5 is based on https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-fastai-model-with-torchserve/

It depends on having already generated a Torchscript model file from the MDv5 weights ([available here](https://github.com/microsoft/CameraTraps/releases/tag/v5.0)) with [yolov5's export.py function](https://github.com/ultralytics/yolov5/commit/6dd6aea0866ba9115d38e2989f59cf1039b3c9d2). Then, a .mar archive needs to be created to package the Torchscript model with a custom handler and model metadata. See animl-ml/api/megadetectorv5/Readme.md for details.

