# Alita v3.03 classifier Deployment Instructions

Alita was trained by Olly Powell (of [Weka Research](https://wekaresearch.com/)) for the New Zealand Department of Conservation, and its full repository be found [here](https://github.com/Wologman/Alita).

The full Alita pipeline is an ensemble comprised of MegaDetector, an 81-species classsifer, and a final heuristic decision making step. However, the instructions below are for deploying just the species classifier as a stand-alone endpoint for use in Animl.

The classifier is an EfficientNet v2 model, trained in PyTorch (Lightning), that classifies 81 species found in New Zealand. It expects an input of `(480, 480)`.

The following instructions are for deploying this PyTorch model to a Sagemaker Serverless Endpoint served in a Torchserve container. In order to create and deploy the model archive from scratch, we need to work across two different environments:

1. your local environment, where you will:

   1. download the model weights and class list from s3
   2. load the model weights into PyTorch and re-compile to torchscript for CPU
   3. create a .mar Model Archive file for Torchserve
   4. [Optionally] test the model and handler in a torchserve Docker container by building it and requesting inference locally
   5. Upload the model archive to s3

2. The Sagemaker notebook environment where you will:
   1. download the .mar archive from s3
   2. build the deploy image and push to ECR
   3. create a serverless endpoint configuration
   4. deploy and test a serverless endpoint

## 1. Download and unzip model checkpoint and classes

From this directory, run:

```bash
aws s3 sync s3://animl-model-zoo/alitav3/ .
```

You should have a directory structure that looks like:

```
...
/alitav3
    |-- exported-model
        |-- alitav3_compiled_cpu.pt
        |-- alitav3.mar
        |-- index_to_name.json
    |-- original-model
        |--Exp_60_run_01_best_weights.pt
        |--Exp_60_run_01_class_names.json
        |--Exp_60_run_01.yaml
        |--taxon-mapping.csv
    ...
```

> **NOTE:** if there's a `.mar` file present in the `/exported-model` directory, that's the older/current Torchscript Model Archive, and unless you want to re-compile this model for CPU and create a new `.mar` file (perhaps because the weights or the inference code changed), you can skip to the [Locally build/test step](#locally-build-serve-and-test-the-torchscript-model-with-torchserve).

## 2. Prepare the Model for Serving

Create and activate a Conda environment and install dependencies by running the following form this directory:

```bash
conda create -n alitav3 python=3.11 pip -y
conda activate alitav3
pip install -r requirements.txt
```

Next, step through `alitav3_compile.ipynb` to load the weights into PyTorch locally and re-compile to Torchserve for CPU. The notebook should produce a torchscript model `alitav3_compiled_cpu.pt` in the `./exported-model/` directory.

Finally, step through `alitav3_prep_class_list.ipynb` to convert the class names to Torchserve format. The notebook will generate a `exported-model/index_to_name.json` file from the original class names.

## 3. Create Model Archive and Build Container

Run the automated build script:

```bash
chmod +x build_container.sh
./build_container.sh
```

This script will:

- Install `torch-model-archiver`
- Create the model archive (.mar file)
- Build the Docker image

Finally, upload the exported files to S3 so that they are accessible for deployment and future use:

```bash
aws s3 cp --recursive ./exported-model s3://animl-model-zoo/alitav3/exported-model
```

## 4. Test the Container Locally

Start the container locally:

```bash
bash docker-run.sh $(pwd)/exported-model
```

Once the container is running, test it with a sample image:

```bash
python test_inference.py ./tests/test-data/stoat-test.jpg --bbox 0.48077553510665894 0.3209689259529114 0.690616250038147 0.5912476778030396
```

```bash
python test_inference.py ./tests/test-data/kea-test.jpg --bbox 0.5261203050613403 0.00028116704197600484 0.9672221541404724 0.40589991211891174
```

```bash
python test_inference.py ./tests/test-data/rat-test.jpg --bbox 0.4617251753807068 0.01082997303456068 0.6275747418403625 0.3216787576675415
```

Expected results:

```bash
# expected predictions for stoat-test.jpg
=== Prediction Results ===
Top 5 predictions:
 1. stoat                0.9944
 2. weasel               0.0012
 3. ferret               0.0007
 4. possum               0.0002
 5. rat                  0.0001

# expected predictions for rat-test.jpg
=== Prediction Results ===
Top 5 predictions:
 1. rat                  0.9226
 2. mouse                0.0294
 3. possum               0.0019
 4. hedgehog             0.0014
 5. stoat                0.0008

# expected preditions for kea-test.jpg
=== Prediction Results ===
Top 5 predictions:
 1. kea                  0.9986
 2. mouse                0.0017
 3. weka                 0.0002
 4. blackbird            0.0002
 5. kaka                 0.0000
```

## 5. Deployment to SageMaker

For production deployment to AWS SageMaker Serverless Inference:

1. Start a SageMaker Notebook instance
2. Clone this repository
3. Run the deployment notebook: `alitav3_deploy.ipynb`

The deployment notebook will:

- Build and push the Docker image to ECR
- Create SageMaker model and endpoint configurations
- Deploy batch and real-time serverless endpoints
- Test the deployed endpoints
