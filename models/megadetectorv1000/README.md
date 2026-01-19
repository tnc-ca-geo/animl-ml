# MegaDetector SageMaker Handler

This handler serves the MegaDetector v1000 (mdv1000.0.0-redwood) model for inference via FastAPI.

## Docker Build and Run

### Build the Docker image
```bash
docker buildx build --platform linux/amd64 -t megadetector-v1000 .
```

### Run locally
```bash
docker run -p 8080:8080 megadetector-v1000
```

### Test the endpoint
```bash
# Health check
curl http://localhost:8080/ping

# Run inference
curl -X POST http://localhost:8080/invocations \
  --data-binary @image.jpg
```

## Processing Pipeline

The MegaDetector PTDetector handles preprocessing and post-processing internally:

### 1. Image Preprocessing
**Location:** [pytorch_detector.py#L1177-L1183](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/detection/pytorch_detector.py#L1177-L1183)

When you pass a PIL Image or numpy array to `generate_detections_one_image()`, the detector automatically:
- Calls `preprocess_image()` internally
- Applies letterboxing to resize the image while preserving aspect ratio
- Normalizes pixel values
- Converts to the appropriate tensor format

### 2. Non-Maximum Suppression (NMS)
**Location:** [pytorch_detector.py#L1305-L1314](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/detection/pytorch_detector.py#L1305-L1314)

After model inference, NMS is automatically applied to filter overlapping detections:
- **IoU threshold:** 0.45 for "classic" compatibility mode, 0.6 otherwise
- Uses either the YOLO library's `non_max_suppression()` or the detector's custom `nms()` method
- Filters detections before they are returned in the result dict

## Output Format

The handler returns a list containing detections for the image (matching MDv5 format):

```python
[[
    {
        'x1': 0.188,        # Normalized x coordinate of top-left corner (0-1)
        'y1': 0.6101,       # Normalized y coordinate of top-left corner (0-1)
        'x2': 0.3398,       # Normalized x coordinate of bottom-right corner (0-1)
        'y2': 0.8401,       # Normalized y coordinate of bottom-right corner (0-1)
        'confidence': 0.894,  # Confidence score
        'class': 1          # Class ID (1=animal, 2=person, 3=vehicle)
    }
]]
```

When there are no detections, returns:
```python
[[]]
```

## Usage

The handler accepts base64-encoded or raw image bytes via POST to `/invocations`:

```python
import requests
import base64

with open('image.jpg', 'rb') as f:
    image_bytes = f.read()

response = requests.post(
    'http://localhost:8080/invocations',
    data=base64.b64encode(image_bytes)
)

detections = response.json()  # Returns [[{...}, {...}]] or [[]]
```
