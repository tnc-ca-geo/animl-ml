# Classifier training resources
Guidance for training classifiers using Animl data.

## `Training a Classifier using SageMaker Studio Lab`

### Prerequisites

Sign up for an [AWS SageMaker Studio Lab](https://studiolab.sagemaker.aws) account, start the runtime, and open your project. SageMaker Studio Lab is a free tool that allows users to train models for a limited time (8 hrs per day) on AWS-hosted GPUs (Tesla T4s).

### Setup

This workflow relies heavily on the [classifier training](https://github.com/microsoft/CameraTraps/tree/main/classification) instructions and code published by Microsoft's AI for Earth Team. The instructions below simply provide guidance on  exporting images and annotations from Animl and shoe-horning them into the AI 4 Earth team's classifier training workflow. As such, we highly recommend referencing the `microsoft/CameraTraps/classification` README.md in conjunction with these instructions, as they go into much more detail about each of the steps outlined here.

#### Clone relevant repos and build Conda environments

To clone the necessary repos, navigate to the `studio-lab-user`'s `home` directory, and from the sidebar menu select the Git icon > "Clone A Repository". Enter the Git repo URL for the [microsoft/CameraTraps](https://github.com/microsoft/CameraTraps) repo (https://github.com/microsoft/CameraTraps.git), and click "Clone". If you had the "Search for environment.yml and build Conda environment." Box checked, it should automatically build the `cameratraps` Conda environment.

Follow the same steps to clone the
- [microsoft/ai4eutils](https://github.com/microsoft/ai4eutils) repo
- [animl-analytics](https://github.com/tnc-ca-geo/animl-analytics) repo
- and this ([animl-ml](https://github.com/tnc-ca-geo/animl-ml)) repo

Next, navigate to `~/Cameratraps/` project root directory and run `conda env update -f environment-classifier.yml --prune` to build the `cameratraps-classifier` Conda environment, which is the primary one we'll be using.

Finally, activate the `cameratraps-classifier` env and install `azure-cosmos` dependency (it's required but seemed to be missing from the env):

```
conda activate cameratraps-classifier
conda install -n cameratraps-classifier -c conda-forge azure-cosmos
```

#### Add additional directories

Add additional directories (`~/classifier-training`, `~/images`, `~/crops`, etc.) so that the contents of your `home/` directory matches the following structure:

```
ai4eutils/                      # Microsoft's AI for Earth Utils repo

animl-analytics/                # animl-analytics repo (utilities for exporting images)

animl-ml/                       # This repo, contains Animl-specific utilities

CameraTraps/                    # Microsoft's CameraTraps repo
    classification/
        BASE_LOGDIR/            # classification dataset and splits
            LOGDIR/             # logs and checkpoints from a single training run

classifier-training/            
    mdcache/                    # cached "MegaDetector" outputs
        v5.0b/                  #   NOTE: MegaDetector is in quotes because we're
            datasetX.json       #   also storing Animl annotations here too
    megaclassifier/             # files relevant to MegaClassifier

crops/                          # local directory to save cropped images
    datasetX/                   # images are organized by dataset
        img0___crop00.jpg

images/                         # local directory to save full-size images
    datasetX/                   # images are organized by dataset
        img0.jpg

```

#### Setup Env variables
The following environment variables are useful to have in `.bashrc`:

```bash
# Python development
export PYTHONPATH="/home/<user>/CameraTraps:/home/<user>/ai4eutils"
export MYPYPATH=$PYTHONPATH
```

It's also helpful to set a `$BASE_LOGDIR` variable for the session:
```bash
export BASE_LOGDIR=/home/studio-lab-user/CameraTraps/classification/BASE_LOGDIR
```

### Export data from Animl and load into SageMaker Studio Lab environment
This will be a two step process:
1. Export annotations and image metadata as COCO for Camera Traps .json document:
  - TODO: link to documentation on how to do that
  - rename file to `<dataset_name>_cct.json` and upload to `~/classifier-training/mdcache/v5.0b/`
2. Copy images listed in `<dataset_name>_cct.json` from S3 to SageMaker env:
  - Make sure you have AWS credentials to read from the `animl-images-archive-prod` bucket
  - Install AWS CLI , boto3, and configure with your AWS credentials (NOTE: creds must be stored in a named profile called `animl`):
  ```bash
  conda install --name cameratraps-classifier -c anaconda boto3
  conda install --name cameratraps-classifier -c conda-forge awscli
  aws configure --profile animl
  ```
  - To download all the images referenced in the cct.json file, navigate to `~/animl-analytics/` and run:
  ```bash
  python utils/download_images.py \
   --coco-file  ~/classifier-training/mdcache/v5.0b/<dataset_name>_cct.json\
   --output-dir ~/images/<dataset_name>
  ```

### Convert exported COCO file to MegaDetector results format
Some of the following steps expect the image annotations to be in the same [format](https://github.com/microsoft/CameraTraps/tree/main/api/batch_processing/#batch-processing-api-output-format) that MegaDetector outputs after processing a batch of images. To convert the COCO for Cameratraps file that we exported from Animl to a MegaDetector results file, navigate to the `/home/studio-lab-user/` directory and run:

```bash
python animl-ml/classification/utils/cct_to_md.py \
  --input_filename ~/classifier-training/mdcache/v5.0b/<dataset_name>_cct.json \
  --output_filename ~/classifier-training/mdcache/v5.0b/<dataset_name>_md.json
```

### Crop images 
To crop images to their detections' respective bounding boxes, run:

```bash
python animl-ml/classification/utils/crop_detections.py \
    ~/classifier-training/mdcache/v5.0b/<dataset_name>_md.json \
    ~/crops/<dataset_name> \
    --images-dir ~/images/<dataset_name> \
    --threshold 0 \  # irrelevant for ground-truthed detections but we pass it in anyhow
    --square-crops \
    --threads 50 \
    --logdir $BASE_LOGDIR
```

### Convert MegaDetector results file to queried_images.json
Microsoft's `CameraTraps/classification/create_classification_dataset.py` takes the output of `json_validator.py` (see their docs on what that does [here](https://github.com/microsoft/CameraTraps/tree/main/classification#2-query-megadb-for-labeled-images)) as an input. To convert our MegaDetecotr results file to `queried_images.json` file, run: 

```bash
python animl-ml/classification/utils/md_to_queried_images.py \
  --input_filename ~/classifier-training/mdcache/v5.0b/<dataset_name>_md.json \
  --dataset <dataset_name> \
  --output_filename $BASE_LOGDIR/queried_images.json
```

### Create classification dataset & split crops into train/val/test sets
This step is well documented in the `microsoft/CameraTraps/classification` [README](https://github.com/microsoft/CameraTraps/tree/main/classification#4-create-classification-dataset-and-split-image-crops-into-trainvaltest-sets-by-location), but some sample arguments are below: 

```bash
python CameraTraps/classification/create_classification_dataset.py \
    $BASE_LOGDIR \
    --mode csv splits \
    --queried-images-json $BASE_LOGDIR/queried_images.json \
    --cropped-images-dir ~/crops \
    --detector-output-cache-dir ~/classifier-training/mdcache --detector-version 5.0b \
    --threshold 0 \
    --min-locs 3 \
    --val-frac 0.2 --test-frac 0.2 \
    --method random
```

### (Optional) inspect dataset
Follow instructions [here](https://github.com/microsoft/CameraTraps/tree/main/classification#5-optional-manually-inspect-dataset), but add and run the following code block at the beginning of the "Imports and Constants" section of `inspect_dataset.ipynb`:

```python
import sys
sys.path.append("/home/studio-lab-user/CameraTraps")
sys.path.append("/home/studio-lab-user/stuiai4eutils")
sys.path
```

### Train classifier

```bash
python train_classifier.py \
    $BASE_LOGDIR \
    ~/crops \
    --model-name efficientnet-b3 --pretrained \
    --label-weighted \
    --epochs 50 --batch-size 160 --lr 3e-5 \
    --weight-decay 1e-6 \
    --num-workers 4 \ # default is 8, but I got warnings that the max was 4 in SageMaker Studio Lab env
    --logdir $BASE_LOGDIR --log-extreme-examples 3
```

NOTE: I ran into a few issues running the command above: 
- had to update torchvision and pytorch:
```bash
conda update torchvision
conda update pytorch
```
- The environment initially had trouble finding CUDA, but [trick described here for Linux](https://github.com/microsoft/CameraTraps/tree/main/classification#verifying-that-cuda-is-available-and-dealing-with-the-case-where-it-isnt) solved it. 
- After those fixes I was able to get the training started but quickly ran into a `RuntimeError: CUDA out of memory.` error. The SageMaker Studio Lab env gives you 15GB memory, which evidently was not enough, but I was able to resume training by dropping the `--batch-size` param down to 32. 
- Ultimately, however, after 10 epochs, I maxed out SageMaker Studio Lab's 25GB of disk space. The vast majority of the disk usage was from Conda envs and packages (22GB).