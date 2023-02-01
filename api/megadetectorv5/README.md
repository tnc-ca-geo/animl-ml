# Setup Instructions

## Download weights and torchscript model
From this directory, run:
```
aws s3 sync s3://animl-model-zoo/megadetectorv5/ models/megadetectorv5/
```

## Export yolov5 weights as torchscript model

First, clone and install yolov5 dependencies and yolov5 following these instructions: https://docs.ultralytics.com/tutorials/torchscript-onnx-coreml-export/

Then, if running locally, make sure to install the correct version of torch and torchvision, the same versions used to save the torchscript megadetector model, we need to use these to load the torchscript model. Check the Dockerfile for versions.

Size needs to be same as in mdv5_handler.py for good performance. Run this from this directory 
```
python ../../../yolov5/export.py --weights models/megadetectorv5/md_v5a.0.0.pt --img 1280 1280 --batch 1 
```
this will create models/megadetectorv5/md_v5a.0.0.torchscript 

## Run model archiver
first, `pip install torch-model-archiver` then,

```
torch-model-archiver --model-name mdv5 --version 1.0.0 --serialized-file models/megadetectorv5/md_v5a.0.0.torchscript --extra-files index_to_name.json --handler mdv5_handler.py
mkdir -p model_store
mv mdv5.mar model_store/megadetectorv5-yolov5-1-batch-1280-1280.mar
```

The .mar file is what is served by torchserve.

## Serve the torchscript model with torchserve

```
docker build -t cv2-torchserve:0.5.3-cpu
bash docker_mdv5.sh
```

Note: The Dockerfile is adapted from the base pytorch torchserve image in order to address an issue with the libjpeg version. We use conda to install dependencies, including torchserve, because conda installs the version of libjpeg that was used to train and test Megadetector originally. See this issue for more detail https://github.com/pytorch/serve/issues/2054

## Return prediction in normalized coordinates with category integer and confidence score

```
curl http://127.0.0.1:8080/predictions/mdv5 -T ../../input/sample-img-fox.jpg
```

# Deploying the model to a Sagemaker Serverless Endpoint

Once you have run the model archiver step above, you're ready to upload that model to s3 so it can be deployed to a serverless inference endpoint!

Run 

```
aws s3 cp model_store/megadetectorv5-yolov5-1-batch-1280-1280.mar s3://animl-model-zoo/megadetectorv5/megadetectorv5-yolov5-reproduced.mar
``` 

to copy the model to the appropriate s3 bucket where pytorch and tensorflow models (for MIRA) are stored.

You'll also need to push the locally built docker image to the ECR repository (which if it is not created, you can create in the deploy notebook).

```
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 830244800171.dkr.ecr.us-w
est-2.amazonaws.com

docker tag cv2-torchserve:0.5.3-cpu 830244800171.dkr.ecr.us-west-2.amazonaws.com/torchserve-mdv5-sagemaker:latest

docker push 830244800171.dkr.ecr.us-west-2.amazonaws.com/torchserve-mdv5-sagemaker:latest
```

then open the jupyter notebook titled mdv5_deploy.ipynb from a Sagemaker Notebook instance. You can also run this locally but would need to set up sagemaker dependencies so the notebook instance is recommended.
