# Small Animal Classifier - FastAPI Handler Implementation

## Overview
This document describes how to implement a FastAPI handler for the small-animal-classifier that accepts image inputs in MegaDetector format and outputs predictions as specified in the MegaDetector v5 output format.

## Key Components

### Input Format:
- Single image file (path-based input)
- Output in MegaDetector v5 JSON format 
- Each image gets a single synthetic whole-image "detection" with category `object` and box `[0,0,1,1]`
- Top-N predictions attached as class-confidence pairs

### Output Format:
Follows MegaDetector results format with synthetic whole-image detection:
```json
{
  "images": [
    {
      "file": "image.jpg",
      "detections": [
        {
          "category": "object", 
          "bbox": [0, 0, 1, 1],
          "confidences": [
            {"class": "bird", "confidence": 0.9543},
            {"class": "squirrel", "confidence": 0.8721}
          ]
        }
      ],
      "failure": false
    }
  ]
}
```

## Implementation Steps

### 1. Load Model and Checkpoint Metadata
- Use `strip_checkpoint.py` logic to load `.stripped.pt` files
- Extract class names, model name, image size, normalization parameters
- Create model with `timm.create_model()` using extracted metadata
- Handle preprocessing using `ValTransform` from `transforms.py`

### 2. Preprocessing Pipeline 
- Load and validate input image using PIL
- Apply banner cropping and resizing using `ValTransform.crop_banner` and `resize`
- Convert to tensor and normalize using extracted normalization parameters
- Match training preprocessing pipeline exactly

### 3. Inference Processing
- Run model inference with loaded weights
- Apply softmax to output logits 
- Format predictions as class-confidence pairs sorted by confidence
- Limit to top-N classifications (default 3)

### 4. Output Formatting
- Wrap results in MegaDetector v5 format structure
- Include synthetic whole-image detection with box `[0,0,1,1]`
- Use "object" category for all detections
- Format confidences as 4-decimal places

## Required Files and Code References

- `src/run_inference.py`: Contains `load_model()` (lines 191-215), `infer()` function, output formatting
- `src/transforms.py`: Contains `ValTransform` class that implements preprocessing pipeline  
- `src/strip_checkpoint.py`: Shows how stripped checkpoints are structured with model metadata

## FastAPI Handler Structure 
```python
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, Request
import json

app = FastAPI()

# Global model state initialization (like in existing handlers)
model, classes, transform, img_size = None, None, None, None

# Load model in main function or on startup
def load_model_from_checkpoint(checkpoint_path):
    # Uses logic from strip_checkpoint.py to extract metadata
    # Uses timm.create_model() to create model 
    # Applies ValTransform preprocessing
    pass

@app.post("/invocations")
async def invoke(request: Request):
    # Parse request - decode image from base64 or file path
    # Apply preprocessing with ValTransform  
    # Run inference through model
    # Format output as MegaDetector v5 JSON with synthetic detection
    pass

@app.get("/ping")
async def ping():
    return {"status": "Healthy"}
```