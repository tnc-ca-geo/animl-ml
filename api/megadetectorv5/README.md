# Setup Instructions

In order to create and deploy the Megadetector model archive from scratch, we need to work across two different environments

1. your local environment, where you will:
   1. Download the model weights from s3
   2. Install dependencies for `torch-model-archiver`
   3. Run `torch-model-archiver` to generate the .mar archive
   4. Upload the model archive to s3

2. The Sagemaker notebook environment where you will:
   1. download the .mar archive
   2. build the deploy image and push to ECR
   3. create a serverless endpoint configuration
   4. deploy and test a serverless endpoint

## Download weights and recreate ONNX model file

From this directory, run:
```
aws s3 sync s3://animl-model-zoo/mdv5-weights-models/ model-weights
```
TODO revise readme to cover using torchscript + Intel IPEX if it works out faster than ONNX

If you want to quickly run model inference outside of the deployment environment without post processing steps, you can use the ONNX model file.

We'll use the model weights, 'md_v5a.0.0.pt', to recreate the ONNX model file and package the ONNX model file in a .mar archive. Or, you can skip this step and use the ONNx model file in the model-weights directory. Torchserve has native support for ONNX and other compiled model formats: https://github.com/pytorch/serve/blob/master/docs/performance_guide.md

You can use this python environment, which mirrors the Dockerfile dependencies and cna be used to run the testing notebooks in this directory (not the deploy notebook):

`conda create -n mdv5a python=3.9`

```
conda activate mdv5a

pip install "gitpython" "ipython" "matplotlib>=3.2.2" "numpy==1.23.4" "opencv-python==4.6.0.66" \
"Pillow==9.2.0" "psutil" "PyYAML>=5.3.1" "requests>=2.23.0" "scipy==1.9.3" "thop>=0.1.1" \
"torch==1.10.0" "torchvision==0.11.1" "tqdm>=4.64.0" "tensorboard>=2.4.1" "pandas>=1.1.4" \
"seaborn>=0.11.0" "setuptools>=65.5.1" "intel-extension-for-pytorch", "torch-model-archiver", "httpx"
```
then, run the export step

```
python yolov5/export.py --imgsz '(960,1280)' --weights model-weights/md_v5a.0.0.pt --include torchscript
mv model-weights/md_v5a.0.0.torchscript model-weights/md_v5a.0.0.960.1280.torchscript
```

Note, we are using the yolov5 source when compiling the model for deployment. This is [yolov5 commit hash 5c91da](https://github.com/ultralytics/yolov5/tree/5c91daeaecaeca709b8b6d13bd571d068fdbd003)


this will create models/megadetectorv5/md_v5a.0.0.onnx and move it to the correct directory for local testing. It will expect a fixed image size input of 960 x 1280 and one image at a time (batch size of 1). Large images will be scaled down to fit this size and padded to preserve aspect ratio. Small images will not be scale dup, and will instead be padded to preserve aspect ratio and not change resolution.


# Creating the model archive

`pip install torch-model-archiver` then,

```
torch-model-archiver --model-name mdv5a --version 1.0.0 --serialized-file model-weights/md_v5a.0.0.960.1280.torchscript --extra-files index_to_name.json --handler mdv5_handler.py
mv mdv5a.mar model_store/mdv5a.mar
```

The .mar file is what is served by torchserve on the serverless endpoint and includes the handler code that processes image requests and the ONNX model file that has traced and compiled the model weights defining what the Megadetector model has learned and the model structure defined by the yolov5 code.

We can locally test this model prior to deploying.

## Locally build and serve the ONNX model with torchserve

```
docker build -t torchserve-mdv5a:0.5.3-cpu .
bash docker_mdv5.sh $(pwd)/model_store
```

## Return prediction in normalized coordinates with category integer and confidence score

The torchserve endpoint can be queried like
```
curl http://127.0.0.1:8080/predictions/mdv5 -T ../../input/sample-img-fox.jpg
```

However, to test the endpoint that is queried during production, test the sagemaker endpoint, which uses the configurations set in deployment/config.properties to adjust threads, worker count, and other container parameters:

```
curl http://127.0.0.1:8080/invocations -T ../../input/sample-img-fox.jpg
```

Note: In the past we attempted to adapt the Dockerfile to address an issue with the libjpeg version. We used conda to install dependencies, including torchserve, because conda installs the version of libjpeg that was used to train and test Megadetector originally. See this issue for more detail https://github.com/pytorch/serve/issues/2054. We [reverted this change](https://github.com/tnc-ca-geo/animl-ml/pull/98/commits/b2bbff5316fbb15023025b2373dcdc9354dd26a7) because installing from conda ballooned the image size above the 10Gb limit set by Sagemaker Serverless. The confidence results are virtually equivalent with the different libjpeg version. See the local_ts_inf_compare.ipynb at the bottom for that exploration. 

Also, see debug_single_img_inference.ipynb for a notebook walktrough of single image inference and plotting bbox results. This also shows querying the container from the notebook and plotting the result.

# Deploying the model to a Sagemaker Serverless Endpoint

Once you have run the model archiver step above, you're ready to upload that model to s3 so it can be deployed to a serverless inference endpoint!

Run 


`aws s3 cp model_store/mdv5a.mar s3://animl-model-zoo/`

to copy the model to the appropriate s3 bucket where pytorch and tensorflow models (for MIRA) are stored.

You'll also need to push the locally built docker image to the ECR repository. Since the images are large, it is fastest to do this from a sagemaker notebook instance in the mdv5_deploy.ipynb. Instructions are below if you want to do this from your local machine anyway.

```
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 830244800171.dkr.ecr.us-w
est-2.amazonaws.com

docker tag cv2-torchserve:0.5.3-cpu 830244800171.dkr.ecr.us-west-2.amazonaws.com/torchserve-mdv5-sagemaker:latest

docker push 830244800171.dkr.ecr.us-west-2.amazonaws.com/torchserve-mdv5-sagemaker:latest
```

then open the jupyter notebook titled mdv5_deploy.ipynb from a Sagemaker Notebook instance. You can also run this deploy notebook locally but would need to set up dependencies so the notebook instance is recommended.
