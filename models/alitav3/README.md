# Alita v3 Classifier Deployment Instructions

The Alita v3 classifier is a PyTorch model for New Zealand species classification, originally developed by the Wologman team. The original repository can be found [here](https://github.com/Wologman/Alita).

It classifies wildlife species from camera trap images into 79 New Zealand species categories and expects an input of `(480, 480)`.

## Prerequisites

1. Compiled model weights at `exported-model/alitav3_compiled_cpu.pt`
2. Docker installed and running
3. AWS CLI configured (for deployment)
4. Python environment with required dependencies

## Quick Start

### 1. Prepare the Model for Serving

First, run the preparation notebook to convert class names to TorchServe format:

```bash
jupyter notebook alitav3_prepare_serving.ipynb
```

This will create `exported-model/index_to_name.json` from the original class names.

### 2. Build and Test Locally

Run the automated build and test script:

```bash
chmod +x build_and_test.sh
./build_and_test.sh
```

This script will:

- Install `torch-model-archiver`
- Create the model archive (.mar file)
- Build the Docker image
- Start the container locally

### 3. Test the Model

Once the container is running, test it with a sample image:

```bash
python test_inference.py /path/to/test/image.jpg --bbox 0 0 1 1
```

Or use curl directly:

```bash
# Encode your test image
IMG_STRING=$(base64 -i /path/to/test/image.jpg)
BBOX=[0,0,1,1]
PAYLOAD=$( jq -n \
            --arg image "$IMG_STRING" \
            --arg bbox "$BBOX" \
            '{image: $image, bbox: $bbox}' )

# Test the endpoint
curl -i http://127.0.0.1:8080/invocations -F body=$PAYLOAD
```

Expected response format:

```json
{
  "kiwi": 0.8234,
  "possum": 0.1234,
  "cat": 0.0456,
  "rat": 0.0076,
  "mouse": 0.0001
}
```

## Manual Steps

If you prefer to run the steps manually:

### 1. Create Model Archive

```bash
pip install torch-model-archiver

torch-model-archiver \
    --model-name alitav3 \
    --version 3.0.3 \
    --serialized-file exported-model/alitav3_compiled_cpu.pt \
    --extra-files exported-model/index_to_name.json \
    --handler alitav3_handler.py

mv alitav3.mar exported-model/alitav3.mar
```

Finally, upload the exported files to S3 so that they are accessible for deployment and future use:

```bash
aws s3 cp --recursive ./exported-model s3://animl-model-zoo/alitav3/exported-model
```

### 2. Build Docker Image

```bash
docker build -t alitav3:latest-cpu .
```

### 3. Run Container

```bash
bash docker-run.sh $(pwd)/exported-model
```

## Deployment to SageMaker

For production deployment to AWS SageMaker Serverless Inference:

1. Start a SageMaker Notebook instance
2. Clone this repository
3. Run the deployment notebook: `alitav3_deploy.ipynb`

The deployment notebook will:

- Build and push the Docker image to ECR
- Create SageMaker model and endpoint configurations
- Deploy batch and real-time serverless endpoints
- Test the deployed endpoints

## Model Details

- **Input Size**: 480x480 pixels
- **Classes**: 79 New Zealand species
- **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Postprocessing**: Sigmoid activation for probability scores
- **Architecture**: Based on original Alita v3 model

## File Structure

```
alitav3/
├── exported-model/
│   ├── alitav3_compiled_cpu.pt      # Compiled model
│   ├── index_to_name.json           # Class mappings
│   └── alitav3.mar                  # TorchServe model archive
├── original-model/
│   ├── Exp_60_run_01_best_weights.pt    # Original weights
│   ├── Exp_60_run_01_class_names.json   # Original class names
│   └── Exp_60_run_01.yaml               # Model config
├── deployment/
│   ├── config.properties            # TorchServe config
│   └── dockerd-entrypoint.sh        # Container entry point
├── tests/
│   └── test-data/                   # Test images
├── alitav3_handler.py               # Custom TorchServe handler
├── alitav3_prepare_serving.ipynb    # Preparation notebook
├── alitav3_deploy.ipynb             # SageMaker deployment notebook
├── build_and_test.sh               # Automated build script
├── test_inference.py               # Test script
├── docker-run.sh                   # Docker run script
├── Dockerfile                      # Container definition
└── README.md                       # This file
```

## Troubleshooting

### Container Issues

- Ensure Docker is running and you have sufficient memory allocated
- Check container logs: `docker logs $(docker ps -q --filter ancestor=alitav3:latest-cpu)`

### Model Loading Issues

- Verify the compiled model file exists and is not corrupted
- Check that the model was compiled with compatible PyTorch versions

### Inference Issues

- Ensure images are properly base64 encoded
- Verify bounding box coordinates are in relative format [ymin, xmin, ymax, xmax]
- Check that the image format is supported (JPEG, PNG)

### Memory Issues

- The model requires approximately 4GB of memory
- Adjust Docker memory limits if needed
- For SageMaker, ensure sufficient memory allocation in endpoint config

## Stopping the Container

To stop the running container:

```bash
docker stop $(docker ps -q --filter ancestor=alitav3:latest-cpu)
```
