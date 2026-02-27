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
- `bbox`: Bounding box in MegaDetector format `[y1, x1, y2, x2]` (normalized 0-1)

```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "bbox": [0.1, 0.2, 0.5, 0.6]
}
```

## Processing Pipeline

### 1. Image Decoding
- Decodes base64 image to PIL Image

### 2. Bbox Conversion
- Converts from MegaDetector format `[y1, x1, y2, x2]` to cropping format `[x, y, width, height]`

### 3. Image Cropping
Uses Dan Morris's MegaDetector preprocessing method:
- Squares the bounding box (uses max of width/height)
- Adds padding to prevent over-enlargement of small animals
- Centers the detection within the crop
- Pads with black (0) to maintain square aspect ratio

### 4. Classification
- Runs YOLOv8 classification on the cropped image
- Returns probabilities for all 17 classes

## Output Format

The handler returns a JSON object mapping class names to confidence scores:

```json
{
  "bird": 0.001,
  "cat": 0.002,
  "deer": 0.003,
  "dog": 0.001,
  "goat": 0.001,
  "hare": 0.002,
  "hedgehog": 0.001,
  "human": 0.001,
  "mustelid": 0.001,
  "pig": 0.002,
  "possum": 0.950,
  "rabbit": 0.001,
  "rat": 0.001,
  "rodent": 0.001,
  "sheep": 0.001,
  "ship-rat": 0.001,
  "vehicle": 0.001
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

See `nzi-adsv1_deploy.ipynb` for step-by-step deployment instructions.
