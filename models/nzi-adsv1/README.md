# NZI-ADS-v1 SageMaker Handler

Credit to [Peter van Lunteren of Addax Data Science](https://addaxdatascience.com/) and the [New Zealand Department of Conservation](https://www.doc.govt.nz/) for training and open sourcing this classifier.


This handler serves the NZI-ADS-v1 (New Zealand Invasive Species Classifier) model for inference via FastAPI.

## Model Information

- **Model**: New Zealand Invasives v1
- **Framework**: PyTorch (YOLOv8 classification)
- **Input**: 640x640 RGB images (pre-cropped)
- **Classes**: 17 species and taxonomic groups
- **Developer**: Addax Data Science
- **Owner**: New Zealand Department of Conservation
- **License**: CC BY-NC-SA 4.0
- **HuggingFace**: [Addax-Data-Science/NZI-ADS-v1](https://huggingface.co/Addax-Data-Science/NZI-ADS-v1)

## Docker Build and Run

### Build the Docker image
```bash
docker buildx build --platform linux/amd64 -t nzi-adsv1 .
```

### Run locally
```bash
docker run -p 8080:8080 nzi-adsv1
```

### Test the endpoint
```bash
# Health check
curl http://localhost:8080/ping

# Run inference (requires base64-encoded image and bbox)
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64-encoded-image>",
    "bbox": [0.1, 0.2, 0.5, 0.6]
  }'
```

## Request Format

The handler expects a JSON payload with:
- `image`: Base64-encoded image string
- `bbox`: Bounding box in [x, y, width, height] format (normalized 0-1)

```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "bbox": [0.1, 0.2, 0.5, 0.6]
}
```

## Processing Pipeline

### 1. Image Decoding
- Decodes base64 image to PIL Image

### 2. Image Cropping
Uses Dan Morris's MegaDetector preprocessing method:
- Squares the bounding box (uses max of width/height)
- Adds padding to prevent over-enlargement of small animals
- Centers the detection within the crop
- Pads with black (0) to maintain square aspect ratio

### 3. Classification
- Runs YOLOv8 classification on the cropped image
- Returns probabilities for all 17 classes

## Output Format

The handler returns a JSON object mapping class names to confidence scores:

```json
{
    "caprid": 0.99,
    "wallaby": 0.99,
    "rodent": 0.99,
    "lagomorph": 0.99,
    "hedgehog": 0.99,
    "possum": 0.99,
    "sealion": 0.99,
    "mustelid": 0.99,
    "cat": 0.99,
    "dog": 0.99,
    "pig": 0.99,
    "deer": 0.99,
    "cow": 0.99,
    "kea": 0.99,
    "weka": 0.99,
    "kiwi": 0.99,
    "other bird": 0.99
}
```

All 17 classes are returned with their confidence scores (no filtering applied).

## Usage Example

```python
import requests
import base64
import json

# Read and encode image
with open('image.jpg', 'rb') as f:
    image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

# Prepare payload
payload = {
    "image": image_b64,
    "bbox": [0.1, 0.2, 0.5, 0.6]  # [y1, x1, y2, x2] normalized
}

# Send request
response = requests.post(
    'http://localhost:8080/invocations',
    json=payload
)

# Get predictions
predictions = response.json()
print(predictions)
```

## Deployment to SageMaker

Resources are deployed to AWS using the nzi-adsv1_deploy.ipynb notebook.  If you need to redeploy resources, you should manually remove them through the console before stepping through the notebook.

### Add SSM Parameters

Outside of the notebook, create SSM parameters to map endpoint names to config variables in animl-api:

| Parameter Name                            | Parameter Value                                   |
| ----------------------------------------- | ------------------------------------------------- |
| `/ml/nzi-adsv1-batch-endpoint-dev`        | `nzi-adsv1-concurrency-<batch_concurrency>`       |
| `/ml/nzi-adsv1-batch-endpoint-prod`       | `nzi-adsv1-concurrency-<batch_concurrency>`       |
| `/ml/nzi-adsv1-realtime-endpoint-dev`     | `nzi-adsv1-concurrency-<realtime_concurrency>`    |
| `/ml/nzi-adsv1-realtime-endpoint-prod`    | `nzi-adsv1-concurrency-<realtime_concurrency>`    |

### Update animl-api

In the [animl-api](https://github.com/tnc-ca-geo/animl-api) repository:
1. Fetch new SSM parameters
2. Implement model interface in `modelInterfaces.ts`
3. Add request/response handling for NZI-ADS-v1 format

### Add MLModel Record to MongoDB

Create a document in the `mlmodels` collection:
- Model metadata (name, version, endpoint info)
- Array of 17 the `categories` the model returns

### Make Model Available to Projects

Add the new `mlmodel._id` to each Project's `Project.availableMLModels` array to make it available for use in [Automation Rules](https://docs.animl.camera/fundamentals/automation-rules).
