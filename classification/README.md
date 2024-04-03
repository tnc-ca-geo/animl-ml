# Classifier training resources
Guidance for training a classifier using a combination of [Animl](animl.camera/) and [LILA](lila.science/) camera trap data.

## Setup

This workflow draws from utilites from MegaDetector's [classifier training](https://github.com/microsoft/CameraTraps/tree/main/archive/classification) instructions and code originally published by Microsoft's AI for Earth Team but now maintained by Dan Morris. It also uses CV4Ecology's [ct_classifier](https://github.com/CV4EcologySchool/ct_classifier) as a starting point for structuring the project and writing training scritps. For more information on the project structure and best-practices, watch Björn Lütjens' lecture on organizing a classifier training project [here](https://www.youtube.com/watch?v=KAymEcailo0&list=PLGuY5I6wycRghx8ik0OkzHUkeLbyVoQYF&index=3).

### Install dependencies
```bash
### install anaconda 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh 
source .bashrc
```

### Clone relevant repos

Clone the following repos:
- [agentmorris/MegaDetector](https://github.com/agentmorris/MegaDetector)
- [microsoft/ai4eutils](https://github.com/microsoft/ai4eutils)
- [animl-analytics](https://github.com/tnc-ca-geo/animl-analytics)
- and, if you haven't already, this repo: ([animl-ml](https://github.com/tnc-ca-geo/animl-ml))

### Get dependencies
If you're remoting into a computer, it might be worth starting a tmux session for this as creating the conda environment can take a while.
From the `/animl-ml/classification/` directory, run:
```bash
conda env create -f environment-classifier.yml
conda env update -f environment-classifier.yml --prune
conda activate cameratraps-classifier
```

Install pytorch with Pip: 

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

If you have trouble, try uninstalling pytorch with `conda uninstall pytorch` and/or `pip uninstall torch` and reinstalling again with the command above

<!-- Finally, install `azure-cosmos` dependency (it's required but seemed to be missing from the env): /*TODO: do we neeed this?*/

```bash
conda install -n cameratraps-classifier -c conda-forge azure-cosmos
``` -->

### Verify CUDA availability
```bash
### verifying that CUDA is available (and dealing with the case where it isn't) --parallel computing platform 
python ~/MegaDetector/sandbox/torch_test.py

### If CUDA isn't available (the above command returned `CUDA available: False`), please execute the following step:
pip uninstall torch torchvision
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Optional steps for performance
```bash
### Optional steps to make classification faster in Linux
conda install -c conda-forge accimage
```

### Add additional directories 

Add `data/`, subdirectory under `classification/`, and add the following directories below that:
```
classifification/
    classifier/
    data/
        interim/
        processed/
        raw/
    docs/
    notebooks/
    runs/
    utils/
    ...
```

## Download Animl data
This will be a two step process:
1. Export annotations and image metadata as COCO for Camera Traps .json document:
  - documentation on how to export annotations to COCO for Camera Traps from Animl can be found [here](https://docs.animl.camera/fundamentals/export-data)
  - rename file to `<dataset_name>_cct.json` and upload to `~/animl-ml/classification/data/raw/<dataset_name>/<dataset_name>_cct.json` (you sahould also make additinal sub directories in the destination path if they aren't yet there)
2. Download image _files_ listed in `<dataset_name>_cct.json` from S3:
  - Make sure you have AWS credentials to read from the `animl-images-serving-prod` bucket
  - Install AWS CLI , boto3, and configure with your AWS credentials (NOTE: creds must be stored in a named profile called `animl`):
  ```bash
  conda install --name cameratraps-classifier -c anaconda boto3
  conda install --name cameratraps-classifier -c conda-forge awscli
  conda env update --name cameratraps-classifier --file environment-classifier.yml --prune
  aws configure --profile animl
  ```
  - To download all the images referenced in the cct.json file, navigate to `~/animl-analytics/` and run:
  ```bash
  python ~/animl-analytics/utils/download_images.py --coco-file ~/animl-ml/classification/data/raw/animl/animl_cct.json --output-dir ~/animl-ml/classification/data/raw/animl/
  ```

## Download LILA data
TODO: generalize and test steps for downloading and including LILA data
NOTE: these steps have not been tested or updated for the animl-ml/classification repo

```bash
python ./utils/download_lila_dataset_cct.py 
```
### Get list of locations that have rat images ingit them and inspect those locations' distributions of non-rat samples

Step through `/notebooks/find_rat_locations.py`, modifying the config of `download_lila_subset.py` script according to instructions in the notebook.

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

`TODO: UPDATE` figure out how we want to structure `/data/raw/` directory when using multiple datasets and either document merging image directories into one or update image file download workflow to download images to one directory.

## Clean and combine Animl and LILA datasets into single COCO file
Launch and step through the steps in the `clean_and_combine_datasets.ipynb`.

## Crop images 
To crop images to their detections' respective bounding boxes, run:

```bash
python ./utils/crop_detections.py \
    ./data/interim/<dataset_name>/<dataset_name>_clean_cct.json \
    ./data/processed/<dataset_name>/crops \
    --images-dir ./data/raw/<dataset_name> \
    --crop-strategy pad \
    --threads 50 \
    --logdir  ./data/interim/<dataset_name>/logs
```

NOTE: `--crop-strategy` options include: 
- `square` - will pick the longer of the two sides of the bbox, and crop a square around the bbox to that dimension (i.e., there will be more background included in the crop)
- `pad` - will preserve the aspect ratio of the crop but will add padding (pixels with a value of 0) to make the crop square

## Create classification dataset & split crops into train/val/test sets
Preparing a classification dataset for training involves two steps, both of which are performed when running `./utils/create_classification_dataset.py`:
1. Create a CSV file (classification_ds.csv) representing our classification dataset, where each row in this CSV represents a single training example, which is an image crop with its label. Along with this CSV file, we also create a `label_index.json` JSON file which defines a integer ordering over the string classification label names.
2. Split the training examples into 3 sets (train, val, and test). Because camera trap images taken at a single location can often be very similar, it's best-practice to ensure that all samples from a given location are all assigned to the same split, otherwise data leakage may occur. The `create_classification_dataset.py` does this for you by generating 10,000 different train/test/val split combinations in which all samples from a given location are assigned only one split, and then scoring each set based on the following criteria:
    1. the number of examples (labels) for each class roughly matches the desired allocation for each split (e.g. 70%/20%/10%). So if we want to use 70% of the data for training, this scoring function will preference splits in which each individual class has as close to 70% of all available samples of that class as possible, and so on.
    2. for the val and test splits, we're also scoring the degree to which the distribution of classes within the split matches the distribution of classes across the whole dataset (i.e, we're trying to mimic "real-world" speices distributions as best we can).
    3. Additionally, the function preferences splits in which the number of locations in the split that class is present in is also as close to 70% of all locations that the class is present in across the whole dataset.

The splits will be specified in the output `splits.json` file.

```bash
python ./utils/create_classification_dataset.py \
    ./data/interim/<dataset_name> \
    --mode csv cct splits \
    --crops-dir ./data/processed/<dataset_name>/crops \
    --cct-json  ./data/interim/<dataset_name>/<dataset_name>_clean_cct.json \
    --min-locs 3 \
    --val-frac 0.2 --test-frac 0.1 \
    --method random
```

Example args if you just want to re-generate splits:
```bash
python ./utils/create_classification_dataset.py \
    .data/interim/<dataset_name> \
    --mode splits \
    --val-frac 0.2 --test-frac 0.1 \
    --method random
```

### (Optional) inspect dataset
Follow instructions [here](https://github.com/microsoft/CameraTraps/tree/main/archive/classification#5-optional-manually-inspect-dataset), but add and run the following code block at the beginning of the "Imports and Constants" section of `inspect_dataset.ipynb`:

```python
import sys
sys.path.append("/home/<user>/MegaDetector")
sys.path.append("/home/<user>/ai4eutils")
sys.path
```

### Create separate cct files for each split
Step through `./notebooks/create_separate_cct_for_splits.ipynb` to create separate cct files for each split. 

## Train classifier

### Set up Comet for logging
Set up a comet.ml account here: https://www.comet.com/signup and [generate API keys](https://www.comet.com/docs/v2/guides/getting-started/quickstart/#get-an-api-key).

Create a `.env` file in the `./classifier` directory with the following variables:

```
COMET_API_KEY=<Your API Key>
COMET_PROJECT_NAME=<Your Workspace Name>
COMET_WORKSPACE=<Your Project Name>
```

### Set up the training experiement's `run/` directory and config
Copy the sample `config.yml` from `./classification/runs/model-name/experiement-name/config.yml` into a new run directory with the following structure:

```
classifification/
    ...
    runs/
        <model_name>/
            <experiement_name>/
                config.yml
    ...
```

Use the `config.yml` to set/adjust hyperparameters.

TODO: document how to modify the training code to do things not set in `config.yml`, e.g. use a different model architecture, load alternative pretrained weights, use a different loss function, different augmentations, etc.

### Initiate the training script

```bash
python ./classifier/train.py \
    --config ./runs/resnet-18/baseline/config.yml \
    --resume  # use --resume flag to resume training from an existing checkpoint
    # --no-resume # use --no-resume flag to start training from scratch
```

## Generate predictions from a checkpoint

```bash
python ./classifier/predict.py \
    --config runs/resnet-18/baseline/config.yml \
    --checkpoint 200.pt \
    --split val
```

### evaluate results
Run `./notebooks/valuate_results.ipynb`