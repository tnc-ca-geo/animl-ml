"""
Small Animal Classifier model server (ONNX Runtime).

Classification model for small-animal camera traps (California Small Animals
dataset and related). Operates on full frames — no bounding box crop.

Architecture: EVA-02 Large (timm eva02_large_patch14_448), exported to ONNX
from a stripped inference checkpoint (.stripped.pt) produced by the training
repo at https://github.com/agentmorris/small-animal-classifier

Preprocessing pipeline (matches src/transforms.py:ValTransform.__call__,
lines 148-154):
  1. Banner crop  — strip the top/bottom info-bar fractions recorded in the
                    metadata (default ~3% top, 3.5% bottom) to prevent the
                    model from reading timestamps as a classification shortcut.
  2. Square resize — squash/stretch to img_size × img_size (default 448×448)
                    using BILINEAR interpolation, no aspect-ratio preservation.
  3. To float32   — [0,255] uint8 → [0,1] float32, shape (3, H, W)
  4. Normalize    — ImageNet mean/std read from metadata JSON

Ref: src/transforms.py lines 148-154 (ValTransform.__call__)
Ref: src/run_inference.py lines 155-168 (load_model)
Ref: src/run_inference.py lines 248-253 (infer — softmax postprocessing)
"""
import io
import base64
import os
import time
import logging
import json
import numpy as np
from PIL import Image, ImageFile
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
import uvicorn
import onnxruntime as ort

# Don't error on truncated images (common with camera trap photos)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Loading
# =============================================================================

model_path = os.getenv(
    "MODEL_PATH",
    "/opt/ml/model/small-animal-classifier.onnx",
)
metadata_path = os.getenv(
    "METADATA_PATH",
    "/opt/ml/model/small-animal-classifier-metadata.json",
)

try:
    start_time = time.time()
    logger.info(f"Loading metadata from {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    class_names = metadata["classes"]
    img_size = metadata["img_size"]
    norm_mean = np.array(metadata["norm_mean"], dtype=np.float32)
    norm_std = np.array(metadata["norm_std"], dtype=np.float32)
    banner_top = metadata["banner_crop"]["top"]
    banner_bot = metadata["banner_crop"]["bottom"]

    logger.info(f"Loading ONNX model from {model_path}")
    session = ort.InferenceSession(model_path)

    load_time = time.time() - start_time
    logger.info(
        f"Model loaded in {load_time:.2f}s — "
        f"classes={len(class_names)}, img_size={img_size}"
    )

except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess(image: Image.Image) -> np.ndarray:
    """
    Apply the validation preprocessing pipeline from the training repo.

    Steps match src/transforms.py:ValTransform.__call__ (lines 148-154):
      1. Banner crop  — remove top/bottom info-bar fractions
      2. Square resize — squash to img_size × img_size with BILINEAR
      3. To float32   — [0,255] uint8 → [0,1] float32, (3, H, W)
      4. Normalize    — ImageNet mean/std from metadata

    Args:
        image: Full-resolution PIL Image in RGB mode.

    Returns:
        Preprocessed numpy array of shape (1, 3, img_size, img_size).
    """
    # Step 1: Banner crop
    # Remove the top and bottom fractions that typically contain the camera's
    # timestamp / temperature overlay. Prevents the model from using metadata
    # text as a classification shortcut.
    # Ref: src/transforms.py:36-45 (crop_banner)
    # Ref: src/transforms.py:148-150 (ValTransform.__call__)
    w, h = image.size
    top_px = int(round(h * banner_top))
    bot_px = int(round(h * banner_bot))
    if top_px + bot_px < h:
        image = image.crop((0, top_px, w, h - bot_px))

    # Step 2: Square resize (squash, no aspect-ratio preservation)
    # Ref: src/transforms.py:151 (ValTransform.__call__)
    image = image.resize((img_size, img_size), Image.BILINEAR)

    # Step 3: PIL → float32 array [0, 1], shape (H, W, 3)
    # Equivalent to torchvision.transforms.functional.to_tensor
    # Ref: src/transforms.py:152 (ValTransform.__call__)
    arr = np.array(image, dtype=np.float32) / 255.0

    # Step 4: Normalize with mean/std from metadata (ImageNet stats)
    # Equivalent to torchvision.transforms.functional.normalize
    # Ref: src/transforms.py:153 (ValTransform.__call__)
    arr = (arr - norm_mean) / norm_std

    # HWC → CHW (same as to_tensor's channel reordering)
    arr = arr.transpose(2, 0, 1)

    # Add batch dimension → (1, 3, H, W)
    arr = np.expand_dims(arr, axis=0)

    return arr


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# =============================================================================
# FastAPI Server
# =============================================================================

app = FastAPI()


@app.get("/ping")
async def ping():
    """Health check. If model load failed, the container already crashed."""
    return {"status": "Healthy"}


@app.post("/invocations")
async def invoke(request: Request):
    """
    Run Small Animal Classifier inference on a full camera-trap image.

    This model operates on the full frame (no bbox crop). The preprocessing
    pipeline strips the camera info-banner and squashes to a square before
    running the model.

    Args:
        request: JSON body with:
            "image" (str): base64-encoded image

    Returns:
        dict: {class_name: confidence_score} for all classes (no filtering).
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

        # Preprocess: banner crop → square resize → float32 → normalize
        input_arr = preprocess(image)

        # Forward pass
        # logits.float() equivalent: ensure fp32 for softmax precision
        logits = session.run(None, {"input": input_arr})[0].astype(np.float32)

        # Postprocessing: softmax over all classes
        # Ref: src/run_inference.py:248-249 (infer)
        probs = softmax(logits)[0]

        # Format as {class_name: confidence} for all classes
        predictions = {class_names[i]: float(probs[i]) for i in range(len(probs))}

        total_time = time.time() - request_start
        logger.info(f"Request completed in {total_time:.3f}s — {len(predictions)} classes")

        return predictions

    except Exception as e:
        total_time = time.time() - request_start
        logger.error(f"Request failed after {total_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run inference: {e}")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
