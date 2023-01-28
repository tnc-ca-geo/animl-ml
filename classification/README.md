# Classifier training resources
Guidance for training classifiers using Animl data.

## `Training a Classifier using SageMaker Studio Log`

### Prerequisites

Sign up for an [AWS SageMaker Studio Lab](https://studiolab.sagemaker.aws) account, start the runtime, and open your project. SageMaker Studio Lab is a free tool that allows users to train models for a limited time (8 hrs per day) on AWS-hosted GPUs (Tesla T4s).

### Setup

This workflow relies heavily on the [classifier training](https://github.com/microsoft/CameraTraps/tree/main/classification) instructions and code published by Microsoft's AI for Earth Team. Essentially these instructions simply provide guidance on  exporting images and annotations from Animl and shoe-horning them into the AI 4 Earth team's classifier training workflow. As such, we highly recommend referencing the `microsoft/CameraTraps/classification` README.md in conjunction with these instructions, as they go into much more detail about each of the steps outline below.

#### Clone relevant repos and build Conda environments

To clone the necessary repos, navigate to the `studio-lab-user`'s `home` directory, and from the sidebar menu select the Git icon > "Clone A Repository". Enter the Git repo URL for the [microsoft/CameraTraps](https://github.com/microsoft/CameraTraps) repo (https://github.com/microsoft/CameraTraps.git), and click "Clone". If you had the "Search for environment.yml and build Conda environment." Box checked, it should automatically build the `cameratraps` Conda environment.

Follow the same steps to clone the
- [microsoft/ai4eutils](https://github.com/microsoft/ai4eutils) repo
- [animl-analytics](https://github.com/tnc-ca-geo/animl-ml) repo
- and this ([animl-ml](https://github.com/tnc-ca-geo/animl-ml)) repo

Next, navigate to `~/Cameratraps/` project root directory and run `conda env update -f environment-classifier.yml --prune` to build the `cameratraps-classifier` Conda environment, which is the primary one we'll be using for the rest of this workflow.

Finally, activate the `cameratraps-classifier` env and install `azure-cosmos` dependency (it's required but seemed to be missing from the env):

```
conda activate cameratraps-classifier
conda install -c conda-forge azure-cosmos
```

#### Add additional directories

Add additional directories (`~/classifier-training`, `~/images`, `~/crops`, etc.) so that the contents of your `home/` directory matches the following structure:

```
ai4eutils/                      # Microsoft's AI for Earth Utils repo

animl-analytics/                # animl-analytics repo (utilities for exporting images)

animl-ml/                       # This repo

CameraTraps/                    # Microsoft's CameraTraps repo
    classification/
        BASE_LOGDIR/            # classification dataset and splits
            LOGDIR/             # logs and checkpoints from a single training run

classifier-training/            
    mdcache/                    # cached "MegaDetector" outputs
        v5.0b/                  # NOTE: MegaDetector is in quotes because we're
            datasetX.json       # also storing Animl annotations here too
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
export PYTHONPATH="/path/to/repos/CameraTraps:/path/to/repos/ai4eutils"
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
  - To download all the imates referenced in the cct.json file, navigate to `~/animl-analytics/` and run:
  ```bash
  python utils/download_images.py \
   --coco-file  /home/studio-lab-user/classifier-training/mdcache/v5.0b/<dataset_name>_cct.json\
   --output-dir /home/studio-lab-user/images/<dataset_name>/
  ```

### Convert exported COCO file to MegaDetector output format
Many of the following steps expect the image annotations to be in the same format that MegaDetector outputs after processing a batch of images. 

### Crop images 
