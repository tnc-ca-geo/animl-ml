# Classifier training resources
Guidance for training a classifier using a combination of [Animl](animl.camera/) and [LILA](lila.science/) camera trap data.

## `Training a Classifier`

### Setup

This workflow draws from utilites from MegaDetector's [classifier training](https://github.com/agentmorris/MegaDetector/tree/main/classification) instructions and code originally published by Microsoft's AI for Earth Team but now maintained by Dan Morris. It also uses CV4Ecology's [ct_classifier](https://github.com/CV4EcologySchool/ct_classifier) as a starting point for structuring the project and writing training scritps.

#### Install dependencies
```bash
### install anaconda 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh 
source .bashrc
```

#### Clone relevant repos

Clone the following repos:
- [agentmorris/MegaDetector](https://github.com/agentmorris/MegaDetector) /*TODO: do we neeed this?*/
- [microsoft/ai4eutils](https://github.com/microsoft/ai4eutils) /*TODO: do we neeed this?*/
- [animl-analytics](https://github.com/tnc-ca-geo/animl-analytics)
- and, if you haven't already, this repo: ([invasive-animal-detection](git@github.com:CV4EcologySchool/invasive-animal-detection.git))

#### Build and activate Conda environment

```bash
conda create -n classifier
conda env update -f ~/invasive-animal-detection/environment.yml --prune
conda activate classifier
```

Note: if you have issues installing pytorch from the conda `environment.yml` file (i.e., you get a warning similar to `ERROR: No matching distribution found for torch==1.8.1+cu111`), try instaling it with Pip directly: 

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

If you have more trouble, try uninstalling pytorch with `conda uninstall pytorch` and/or `pip uninstall torch` and reinstalling again with the command above

Finally, install `azure-cosmos` dependency (it's required but seemed to be missing from the env): /*TODO: do we neeed this?*/

```bash
conda install -n cameratraps-classifier -c conda-forge azure-cosmos
```

#### Verify CUDA availability
```bash
### verifying that CUDA is available (and dealing with the case where it isn't) --parallel computing platform 
python ~/MegaDetector/sandbox/torch_test.py

### If CUDA isn't available (the above command returned `CUDA available: False`), please execute the following step:
pip uninstall torch torchvision
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Optional steps for performance
```bash
### Optional steps to make classification faster in Linux
conda install -c conda-forge accimage
pip uninstall -y pillow
pip install pillow-simd
```

#### Add additional directories - `TODO: UPDATE` 

Add `data/` subdirectory under `ias-classifier`, and add the following directories below that:
```
invasive-animal-detection/            
    data/
        interim/
        processed/
        raw/
    docs/
    ias-classifier/
    ...
```

Add additional directories - and sub directories listed below (`~/classifier-training`, `~/images`, `~/crops`, etc.) so that the contents of your `home/` directory matches the following structure:

```
ai4eutils/                      # Microsoft's AI for Earth Utils repo

animl-analytics/                # animl-analytics repo (utilities for exporting images)

classifier-training/            
    BASE_LOGDIR/                # classification dataset and splits
        LOGDIR/                 # logs and checkpoints from a single training run

invasive-animal-detection/      # This repo

MegaDetector/                   # MegaDetector repo

crops/                          # local directory to save cropped images
    datasetX/                   # images are organized by dataset
        img0___crop00.jpg

images/                         # local directory to save full-size images
    datasetX/                   # images and metadata are organized by dataset
        images/
            img0.jpg
        metadata/
            datasetX.json

```

#### Setup Env variables `TODO: not sure we need this anymore`
The following environment variables are useful to have in `.bashrc`:

```
# Python development
export PYTHONPATH="/home/<user>/MegaDetector:/home/<user>/ai4eutils"
export MYPYPATH=$PYTHONPATH
```

### Export annotations from Animl and downlaod image files
This will be a two step process:
1. Export annotations and image metadata as COCO for Camera Traps .json document:
  - documentation on how to export annotations to COCO for Camera Traps from Animl can be found [here](https://docs.animl.camera/fundamentals/export-data)
  - rename file to `animl_cct.json` and upload to `~/classifier-training/animl_cct.json`
2. Copy image _files_ listed in `<dataset_name>_cct.json` from S3:
  - Make sure you have AWS credentials to read from the `animl-images-serving-prod` bucket
  - Install AWS CLI , boto3, and configure with your AWS credentials (NOTE: creds must be stored in a named profile called `animl`):
  ```bash
  conda install --name classifier -c anaconda boto3
  conda install --name classifier -c conda-forge awscli
  aws configure --profile animl
  ```
  - To download all the images referenced in the cct.json file, navigate to `~/animl-analytics/` and run:
  ```bash
  python ~/animl-analytics/utils/download_images.py --coco-file  ~/classifier-training/animl_cct.json --output-dir ~/images/animl/
  ```

### Download Island Conservation Cameratraps metadata

```bash
python ./invasive-animal-detection/utils/download_lila_dataset_cct.py 
```
### Get list of locations that have rat images in them and inspect those locations' distributions of non-rat samples

Step through `invastive-animal-detection/notebooks/find_rat_locations.py`, modifying the config of `download_lila_subset.py` script according to instructions in the notebook.

### Download annotations and images from LILA
1. Download AZCopy
```bash
### From home directory
mkdir azcopy
cd azcopy
wget -O azcopy_v10.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy_v10.tar.gz --strip-components=1
```
Add newly created directory to your `$PATH` in `.bashrc`:
```
export PATH=/home/<user>/azcopy:$PATH
```

2. Run `utils/download_lila_subset.py`
```bash
python ./utils/download_lila_subset.py
```

`TODO: UPDATE` figure out how we want to structure `/images` directory and either document merging image directories into one or update image file download workflow to download images to one directory

### Clean and combine Animl and LILA datasets into single COCO file
Launch and step through the steps in the `clean_and_combine_datasets.ipynb`. 

### Crop images 
To crop images to their detections' respective bounding boxes, run:
--crop-strategy options: 
- square (will pick the longer of the two sides of the bbox, and crop a square around the bbox to that dimension)
- pad (will preserve the aspect ratio of the crop but willl add padding (pixels with a value of 0) to make the crop square)

```bash
python ./utils/crop_detections.py \
    ./data/interim/subsample-rats/combined_cct.json \
    ./data/processed/subsample-rats/crops \
    --images-dir ./data/raw/images \
    --crop-strategy square \
    --threads 50 \
    --logdir  ./data/interim/subsample-rats/logs
```

### Create classification dataset & split crops into train/val/test sets
This step is well documented in the `/MegaDetector/classification` [README](https://github.com/agentmorris/MegaDetector/tree/main/classification#4-create-classification-dataset-and-split-image-crops-into-trainvaltest-sets-by-location), but some sample arguments are below:

TODO: Update step instructions to reflect that we can now pass in COCO .json file instead of the `queried_image.json` file.

```bash
python ./utils/create_classification_dataset.py \
    ~/invasive-animal-detection/data/interim/subsample-rats \
    --mode csv cct splits \
    --crops-dir ./data/processed/subsample-rats/crops \
    --cct-json  ./data/interim/subsample-rats/combined_cct.json \
    --min-locs 6 \
    --val-frac 0.2 --test-frac 0.1 \
    --method random
```

Just re-generate splits:
```bash
python invasive-animal-detection/utils/create_classification_dataset.py \
    $BASE_LOGDIR/score-class-dist-2 \
    --mode splits \
    --val-frac 0.2 --test-frac 0.1 \
    --method random
```

### (Optional) inspect dataset
Follow instructions [here](https://github.com/microsoft/MegaDetector/tree/main/classification#5-optional-manually-inspect-dataset), but add and run the following code block at the beginning of the "Imports and Constants" section of `inspect_dataset.ipynb`:

```python
import sys
sys.path.append("/home/<user>/MegaDetector")
sys.path.append("/home/<user>/ai4eutils")
sys.path
```

### Create separate cct files for each split
Step through `invasive-animal-detection/notebooks/create_separate_cct_for_splits.ipynb` to create separate cct files for each split. 

### Train classifier

TODO: add instructions on setting up comet account and getting keys?
TODO: add instructions for creating a `.comet.config` file ([docs](https://www.comet.com/docs/v2/guides/tracking-ml-training/configuring-comet/#configure-comet-using-the-comet-config-file))
Create a `.comet.config` file in the `invasive-animal-detection/ias-classifier` directory with the following variables:

```
[comet]
api_key=<Your API Key>
workspace=<Your Workspace Name>
project_name=<Your Project Name>
```

```bash
python ias-classifier/train.py --config runs/resnet-18/subsample-rats/config.yml
```



```bash
python ias-classifier/predict.py \
    --config runs/resnet-18/baseline/config.yml \
    --checkpoint 200.pt \
    --split val
```

Run `evaluate_results.ipynb`

<!-- ```bash
python MegaDetector/classification/train_classifier.py \
    $BASE_LOGDIR \
    ~/crops \
    --model-name efficientnet-b3 --pretrained \
    --label-weighted \
    --epochs 50 --batch-size 80 --lr 3e-5 \
    --weight-decay 1e-6 \
    --num-workers 8 \
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
- Ultimately, however, after 10 epochs, I maxed out SageMaker Studio Lab's 25GB of disk space. The vast majority of the disk usage was from Conda envs and packages (22GB). -->