# Small Animal Classifier

This model uses a timm EfficientNet B4 backbone with a fine-tuned linear classification head, trained on images from the California Small Animals camera trap dataset.

Model weights and metadata from:
https://github.com/agentmorris/small-animal-classifier

Reference implementation in:
https://github.com/agentmorris/small-animal-classifier/blob/main/src/run_inference.py

## Model Details

- **Architecture**: EfficientNet B4 (timm)
- **Dataset**: California Small Animals camera trap dataset
- **Classes**: 13 animal classes including bird, chipmunk, mouse, squirrel, skink, snake, etc.
- **Input Size**: 448x448 pixels
- **Preprocessing**: Banner cropping (top 3%, bottom 3.5%), resize to input size, normalization

## Deployment

The model is deployed as a FastAPI service using the standard `serve.py` pattern.

### Environment Variables

- `CHECKPOINT_PATH` (optional): Path to the model checkpoint (default: `/opt/ml/model/best_stripped.ckpt`)

### API Endpoints

- `GET /ping` - Health check endpoint
- `POST /invocations` - Run inference on an image

The `/invocations` endpoint expects a base64-encoded image in the request body:
```json
{
  "image": "base64-encoded-image-data"
}
```

And returns results in MegaDetector v5 format with synthetic whole-image detection (category `object`, box `[0,0,1,1]`) containing top-3 class predictions.

## Dockerization

Standard Docker container approach using the opencode ml deploy tool.

For model artifacts:
```bash
# Download the model checkpoint from the releases page
wget https://github.com/agentmorris/small-animal-classifier/releases/download/v1.0/best_stripped.ckpt
```

## Usage Example

```python
import requests
import base64
from PIL import Image
import io

# Load image and encode
image = Image.open('your_image.jpg')
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Send request
response = requests.post(
    "http://localhost:8080/invocations",
    json={"image": img_str}
)
```