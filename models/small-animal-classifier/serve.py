"""
Small Animal Classifier (California Small Animals) classification model server.

This model uses a timm EfficientNet B4 backbone with a fine-tuned linear classification 
head, trained on images from the California Small Animals camera trap dataset.

Model weights and metadata from:
https://github.com/agentmorris/small-animal-classifier

Reference implementation in:
https://github.com/agentmorris/small-animal-classifier/blob/main/src/run_inference.py
"""
import io
import base64
import os
import time
import logging
import json
import pathlib
import platform
from PIL import Image, ImageFile
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
import uvicorn

import torch
import torch.nn.functional as F
import timm
from torchvision import transforms

# Don't error on truncated images (common with camera trap photos)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# The backbone weights may contain Windows-style paths from serialization.
# This alias prevents deserialization failures on Unix/Mac.
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Architecture & Loading
# =============================================================================

def load_model(checkpoint_path, device="cpu"):
    """
    Load a stripped checkpoint and create the model with preprocessing.
    
    Args:
        checkpoint_path: Path to .stripped.pt file
        device: Device to run inference on
        
    Returns:
        tuple: (model, classes, transform, img_size) 
    """
    try:
        start_time = time.time()
        logger.info("Loading checkpoint...")
        
        # Load stripped checkpoint metadata and weights
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # Extract model metadata from checkpoint (as in strip_checkpoint.py)
        classes = ckpt["classes"]
        model_name = ckpt["model_name"]
        img_size = ckpt["img_size"]
        mean = ckpt["norm_mean"] 
        std = ckpt["norm_std"]
        banner_top = ckpt["banner_crop"]["top"]
        banner_bot = ckpt["banner_crop"]["bottom"]
        
        # Create model using timm (matching training setup)
        model = timm.create_model(
            model_name, 
            pretrained=False, 
            num_classes=len(classes)
        )
        # Load weights from checkpoint
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        
        # Build preprocessing pipeline that matches ValTransform from transforms.py
        preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s with {len(classes)} classes")
        
        return model, classes, preprocess, img_size
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# =============================================================================
# Load Model on Startup
# =============================================================================

checkpoint_path = os.getenv(
    "CHECKPOINT_PATH",
    "/opt/ml/model/best_stripped.ckpt",  # Default location in container
)

try:
    model, classes, preprocess, img_size = load_model(checkpoint_path)
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    raise


# =============================================================================
# FastAPI Server
# =============================================================================

app = FastAPI()


@app.get("/ping")
async def ping():
    """Health check. If model load failed, the container already crashed."""
    return {"status": "Healthy"}


def crop_banner(img, top_frac, bot_frac):
    """
    Remove the top/bottom info-banner bands (fractions of height).
    Matches logic from transforms.py.
    """

    w, h = img.size
    t = int(round(h * top_frac))
    b = int(round(h * bot_frac))
    if t + b >= h:
        return img
    return img.crop((0, t, w, h - b))


@app.post("/invocations")
async def invoke(request: Request):
    """
    Run small-animal-classifier inference on an image.
    
    Args:
        request: JSON with 'image' (base64 encoded)
        
    Returns:
        Dictionary containing MegaDetector v5 formatted results with all predictions
    """
    request_start = time.time()
    try:
        # Parse and decode the image
        body = await request.body()
        payload = json.loads(body)
        image_bytes = base64.b64decode(payload["image"])
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply same preprocessing pipeline as training (banner crop + resize + normalize)
        # Crop banner (matches ValTransform behavior but using banner_crop_flag=False for simplicity)  
        image_cropped = crop_banner(image, 0.03, 0.035)
        
        # Resize to model input size and apply normalization
        input_tensor = preprocess(image_cropped)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        # Run inference: logits → softmax → probabilities
        with torch.no_grad():
            output = model(input_batch)
            probabilities = F.softmax(output, dim=1)

        # Format results as MegaDetector v5 format with synthetic detection (category "1" as per run_inference.py)
        scores = probabilities[0].tolist()
        
        # Return ALL predictions, not just top-N
        confidences = []
        for idx, score in enumerate(scores):
            confidences.append([str(idx), round(score, 4)])

        # Wrap result in MegaDetector v5 format with synthetic whole-image detection
        # As per run_inference.py lines 362-364, "category": "1", "bbox": [0,0,1,1]  
        result = {
            "images": [
                {
                    "file": "input.jpg",  # placeholder - would be actual filename
                    "detections": [
                        {
                            "category": "1",
                            "conf": 1.0,
                            "bbox": [0, 0, 1, 1],
                            "classifications": confidences
                        }
                    ],
                    "failure": False
                }
            ]
        }

        total_time = time.time() - request_start
        logger.info(f"Request completed in {total_time:.3f}s - {len(confidences)} classes")

        return result

    except Exception as e:
        total_time = time.time() - request_start
        logger.error(f"Request failed after {total_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run inference: {e}")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()