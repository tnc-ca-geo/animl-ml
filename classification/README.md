# Classifier training resources
Guidance for training a classifier using a combination of [Animl](animl.camera/) and [LILA](lila.science/) camera trap data.

## `Training a Classifier`

### Setup

This workflow draws from utilites from MegaDetector's [classifier training](https://github.com/microsoft/CameraTraps/tree/main/archive/classification) instructions and code originally published by Microsoft's AI for Earth Team but now maintained by Dan Morris. It also uses CV4Ecology's [ct_classifier](https://github.com/CV4EcologySchool/ct_classifier) as a starting point for structuring the project and writing training scritps. For more information on the project structure and best-practices, watch Björn Lütjens' lecture on organizing a classifier training project [here](https://www.youtube.com/watch?v=KAymEcailo0&list=PLGuY5I6wycRghx8ik0OkzHUkeLbyVoQYF&index=3).

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
- [agentmorris/MegaDetector](https://github.com/agentmorris/MegaDetector)
- [microsoft/ai4eutils](https://github.com/microsoft/ai4eutils)
- [animl-analytics](https://github.com/tnc-ca-geo/animl-analytics)
- and, if you haven't already, this repo: ([animl-ml](https://github.com/tnc-ca-geo/animl-ml))

#### Get dependencies
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
```

#### Add additional directories 

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

### Export annotations from Animl and downlaod image files
This will be a two step process:
1. Export annotations and image metadata as COCO for Camera Traps .json document:
  - documentation on how to export annotations to COCO for Camera Traps from Animl can be found [here](https://docs.animl.camera/fundamentals/export-data)
  - rename file to `animl_cct.json` and upload to `~/animl-ml/classification/data/raw/animl/animl_cct.json` (you sahould also make additinal sub directories if they aren't yet there)
2. Copy image _files_ listed in `<dataset_name>_cct.json` from S3:
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

`TODO: generalize and test steps for downloading and including LILA data`
`NOTE: these steps have not been tested or updated for the animl-ml/classification repo`
### Download Island Conservation Cameratraps metadata

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
    ./data/interim/animl/animl_clean_cct.json \
    ./data/processed/animl/crops \
    --images-dir ./data/raw/animl \
    --crop-strategy pad \
    --threads 50 \
    --logdir  ./data/interim/animl/logs
```

### Create classification dataset & split crops into train/val/test sets
Preparing a classification dataset for training involves two steps:
1. Create a CSV file (classification_ds.csv) representing our classification dataset, where each row in this CSV represents a single training example, which is an image crop with its label. Along with this CSV file, we also create a label_index.json JSON file which defines a integer ordering over the string classification label names.
2. Split the training examples into 3 sets (train, val, and test) based on the geographic location where the images were taken. The splits will be specified in the output `splits.json` file. The splits are created by randomly generating 10,000 different potential sets of splits, and then scoring each set based on the following criteria:
- (a) the number of examples (labels) for each class roughly matchs the desired % for each split (e.g. 70%/20%/10%). So if we want to use 70% of the data for training, this scoring function will preference splits in which each individual class has as close to 70% of all available samples of that class as possible.
- (b) for the val and test splits, we're also scoring the degree to which the distribution of classes within the split matches the distribution of classes across the whole dataset (i.e, trying to mimic "real-world") distributions as best we can.
- (c) Additionally, the function preferences splits in which the number of locations in the split that class is present in is also as close to 70% of all locations that the class is present in across the whole dataset.

```bash
python ./utils/create_classification_dataset.py \
    ./data/interim/animl \
    --mode csv cct splits \
    --crops-dir ./data/processed/animl/crops \
    --cct-json  ./data/interim/animl/animl_clean_cct.json \
    --min-locs 3 \
    --val-frac 0.2 --test-frac 0.1 \
    --method random
```

Example args if you just want to re-generate splits:
```bash
python ./utils/create_classification_dataset.py \
    .data/interim/animl \
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

### Set up Comet for logging
Set up a comet.ml account here: https://www.comet.com/signup and [generate API keys](https://www.comet.com/docs/v2/guides/getting-started/quickstart/#get-an-api-key).

Create a `.env` file in the `./classifier` directory with the following variables:

```
COMET_API_KEY=<Your API Key>
COMET_PROJECT_NAME=<Your Workspace Name>
COMET_WORKSPACE=<Your Project Name>
```

### Train classifier
TODO: document run folder structure, config.yml, etc.

```bash
python ./classifier/train.py \
    --config ./runs/resnet-18/animl/config.yml \
    --resume  # use --resume flag to resume training from an existing checkpoint
    # --no-resume # use --no-resume flag to start training from scratch
```

### Generate predictions from a checkpoint

```bash
python ./classifier/predict.py \
    --config runs/resnet-18/animl/config.yml \
    --checkpoint 200.pt \
    --split val
```

### evaluate results
Run `./notebooks/valuate_results.ipynb`