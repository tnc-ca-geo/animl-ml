import io
import base64
import os
import time
import logging
import json
import pathlib
import platform
from PIL import Image, ImageFile, ImageOps
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
import uvicorn
from ultralytics import YOLO
from typing import List

# Don't freak out over truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Make sure Windows-trained models work on Unix
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Init config
model_path = os.getenv(
    "MODEL_PATH",
    "/opt/ml/model/new_zealand_v1.pt",
)


# Load model
try:
    start_time = time.time()
    logger.info(f"Loading model from {model_path}")
    
    model = YOLO(model_path)
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded successfully in {load_time:.2f}s")
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    raise


# Setup FastAPI app
app = FastAPI()


@app.get("/ping")
async def ping():
    """
    Health check endpoint.

    If the model load fails, the container will have already crashed
    """
    return {"status": "Healthy"}


def get_crop(image: Image.Image, bbox: List[float]) -> Image.Image:
    """
    Crop image using Dan Morris's MegaDetector preprocessing method.
    
    Args:
        image: PIL Image (full resolution)
        bbox: Normalized bounding box [x, y, width, height] in range [0.0, 1.0]
    
    Returns:
        Cropped and padded PIL Image ready for classification
    """
    img_w, img_h = image.size
    
    # Denormalize bbox coordinates
    xmin = int(bbox[0] * img_w)
    ymin = int(bbox[1] * img_h)
    box_w = int(bbox[2] * img_w)
    box_h = int(bbox[3] * img_h)
    
    # Square the box (use max dimension)
    box_size = max(box_w, box_h)
    
    # Add padding (prevents over-enlargement of small animals)
    input_size_network = 224
    default_padding = 30
    if box_size >= input_size_network:
        box_size = box_size + default_padding
    else:
        diff_size = input_size_network - box_size
        if diff_size < default_padding:
            box_size = box_size + default_padding
        else:
            box_size = input_size_network
    
    # Center the detection within the squared crop
    xmin = max(0, min(xmin - int((box_size - box_w) / 2), img_w - box_w))
    ymin = max(0, min(ymin - int((box_size - box_h) / 2), img_h - box_h))
    
    # Clip to image boundaries
    box_w = min(img_w, box_size)
    box_h = min(img_h, box_size)
    
    if box_w == 0 or box_h == 0:
        raise ValueError(f"Invalid bbox size: {box_w}x{box_h}")
    
    # Crop and pad to square
    crop = image.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])
    crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)
    
    return crop


@app.post("/invocations")
async def invoke(request: Request):
    """
    Run NZI-ADS-v1 classification on a cropped image.
    """
    request_start = time.time()
    try:
        # Parse JSON body
        decode_start = time.time()
        body = await request.body()
        payload = json.loads(body)
        
        # Decode image
        image_bytes = base64.b64decode(payload['image'])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert bbox from [y1, x1, y2, x2] to [x, y, width, height]
        bbox_input = payload['bbox']
        bbox = [bbox_input[1], bbox_input[0], bbox_input[3] - bbox_input[1], bbox_input[2] - bbox_input[0]]
        
        decode_time = time.time() - decode_start
        logger.info(f"Image decoded in {decode_time:.3f}s - size: {image.size}, bbox: {bbox}")

        # Crop image
        crop_start = time.time()
        crop = get_crop(image, bbox)
        crop_time = time.time() - crop_start

        # Run inference
        # Note: YOLO automatically applies preprocessing transforms internally:
        # 1. CenterCrop to 224x224
        # 2. ToTensor (converts to tensor and normalizes [0,255] -> [0,1])
        # 3. Normalize with mean/std (if specified)
        # See: https://github.com/ultralytics/ultralytics/blob/v8.0.230/ultralytics/engine/predictor.py#L213-L214
        # Transform definition: https://github.com/ultralytics/ultralytics/blob/v8.0.230/ultralytics/data/augment.py#L985-L992
        inference_start = time.time()
        results = model(crop, verbose=False)
        
        # Extract class names and probabilities
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()
        
        inference_time = time.time() - inference_start

        # Format results as {class_name: confidence}
        format_start = time.time()
        predictions = {}
        for idx, class_name in names_dict.items():
            predictions[class_name] = probs[idx]
        format_time = time.time() - format_start

        total_time = time.time() - request_start
        logger.info(
            f"Request completed in {total_time:.3f}s "
            f"(decode: {decode_time:.3f}s, crop: {crop_time:.3f}s, inference: {inference_time:.3f}s, format: {format_time:.3f}s) "
            f"- {len(predictions)} classes"
        )

        return predictions

    except Exception as e:
        total_time = time.time() - request_start
        logger.error(f"Request failed after {total_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run inference: {e}")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
