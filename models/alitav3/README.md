# Alita v3.03 classifier Deployment Instructions

Alita was trained by Olly Powell (of [Weka Research](https://wekaresearch.com/)) for the New Zealand Department of Conservation, and its full repository be found [here](https://github.com/Wologman/Alita).

The full Alita pipeline is an ensemble comprised of MegaDetector, an 81-species classsifer, and a final heuristic decision making step to reconcile predictions. However, the instructions below are for deploying just the species classifier as a stand-alone endpoint.

It's an EfficientNet v2 model, trained in PyTorch (Lightning), that classifies 81 species found in New Zealand. It expects an input of `(480, 480)`.

The following instructions are for deploying this PyTorch model to a Sagemaker Serverless Endpoint served in a Torchserve container. In order to create and deploy the model archive from scratch, we need to work across two different environments:

1. your local environment, where you will:

   1. Download the model weights and class list from s3
   2. load the model weights into PyTorch and re-compile to torchscript for CPU
   3. Install dependencies for `torch-model-archiver` and run `torch-model-archiver` to generate the `.mar` file (a bundled archive that includes the torchscript-compiled model and the hander function)
   4. [Optionally] test the model and handler in a torchserve Docker container by building it and requesting inference locally
   5. Upload the model archive to s3

2. The Sagemaker notebook environment where you will:

   1. download the .mar archive from s3
   2. build the deploy image and push to ECR
   3. create a serverless endpoint configuration
   4. deploy and test a serverless endpoint

## Download and unzip model checkpoint and classes

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
        |--taxon-mapping.csv
    ...
```

> **NOTE:** if there's a `.mar` file present in the `/exported-model` directory, that's the older/current Torchscript Model Archive, and unless you want to re-compile this model for CPU and create a new `.mar` file (perhaps because the weights or the inference code changed), you can skip to the [Locally build/test step](#locally-build-serve-and-test-the-torchscript-model-with-torchserve).

## Load the weights into PyTorch locally and re-compile to Torchserve for CPU

Create and activate a Conda environment and install dependencies by running the following form this directory:

```bash
conda create -n alitav3 python=3.11 pip -y
conda activate alitav3
pip install -r requirements.txt
```

Then step through `alitav3_compile.ipynb`. The notebook should produce a torchscript model 'alitav3_compiled_cpu.pt' in the `./exported-model/` directory.

Note for others using these steps to deploy a different model: the versions of `torch` and `torchvision` that you pin in your `Dockerfile` used for serving must match the versions you use when compiling the model to torchscript. To check which versions you're using in your venv use `pip freeze` and to bump the versions up (or down) use `pip install --upgrade` (e.g. `pip install --upgrade torchvision==0.15.1`).
