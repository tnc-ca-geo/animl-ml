"""
Small Animal Classifier model server.

Credit to Dan Morris (https://github.com/agentmorris) for training and open
sourcing this classifier.

Classification model for small-animal camera traps (California Small Animals
dataset and related). Operates on full frames — no bounding box crop.

Architecture: timm model (eva02_large_patch14_448) rebuilt from a stripped
inference checkpoint (.stripped.pt), as produced by the training repo at
https://github.com/agentmorris/small-animal-classifier

All line number references in this file refer to that repository at commit
835e835 (v1.0 release tag).

Preprocessing pipeline (matches src/transforms.py:ValTransform.__call__,
lines 148-154):
  1. Banner crop  — strip the top/bottom info-bar fractions recorded in the
                    checkpoint (default ~3% top, 3.5% bottom) to prevent the
                    model from reading timestamps as a classification shortcut.
  2. Square resize — squash/stretch to img_size × img_size (default 448×448)
                    using BILINEAR interpolation, no aspect-ratio preservation.
                    Camera faces down → no canonical orientation after banner
                    removal, so squashing is fine.
  3. ToTensor     — PIL uint8 [0,255] → float32 [0,1], shape (3, H, W)
  4. Normalize    — ImageNet mean/std read from checkpoint

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
from PIL import Image, ImageFile
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
import uvicorn

import torch
import torch.nn.functional as F
import timm
import torchvision.transforms.functional as TF

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
    "/opt/ml/model/eva02-20260630-llrd.best.e02-s053514.stripped.pt",
)

try:
    start_time = time.time()
    logger.info(f"Loading model from {model_path}")

    # Load the stripped inference checkpoint.
    # weights_only=False is required because the checkpoint contains Python
    # objects beyond bare tensors (model_name string, classes list, etc.).
    # Ref: src/run_inference.py:155-157 (load_model)
    ck = torch.load(model_path, map_location="cpu", weights_only=False)

    # Rebuild the timm model from the architecture name and class count
    # stored in the checkpoint, then load the stripped state dict.
    # Ref: src/run_inference.py:158-162 (load_model)
    model = timm.create_model(
        ck["model_name"],
        pretrained=False,
        num_classes=ck["num_classes"],
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # Pull all inference-time metadata out of the checkpoint so we don't
    # need to re-read it on every request.
    class_names = ck["classes"]           # list[str], len == num_classes
    img_size = ck["img_size"]             # int, square input side (e.g. 448)
    norm_mean = tuple(ck["norm_mean"])    # (R, G, B) mean for normalize
    norm_std = tuple(ck["norm_std"])      # (R, G, B) std  for normalize
    banner_top = ck["banner_crop"]["top"]     # fraction of height to remove from top
    banner_bot = ck["banner_crop"]["bottom"]  # fraction of height to remove from bottom

    load_time = time.time() - start_time
    logger.info(
        f"Model loaded in {load_time:.2f}s — "
        f"arch={ck['model_name']}, classes={len(class_names)}, img_size={img_size}"
    )

except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


# =============================================================================
# FastAPI Server
# =============================================================================

app = FastAPI()


@app.get("/ping")
async def ping():
    """Health check. If model load failed, the container already crashed."""
    return {"status": "Healthy"}


def preprocess(image: Image.Image) -> torch.Tensor:
    """
    Apply the validation preprocessing pipeline from the training repo.

    Steps match src/transforms.py:ValTransform.__call__ (lines 148-154):
      1. Banner crop  — remove top/bottom info-bar fractions
      2. Square resize — squash to img_size × img_size with BILINEAR
      3. ToTensor     — [0,255] uint8 → [0,1] float32, (3, H, W)
      4. Normalize    — ImageNet mean/std from checkpoint

    Args:
        image: Full-resolution PIL Image in RGB mode.

    Returns:
        Preprocessed tensor of shape (3, img_size, img_size).
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

    # Step 3: PIL → float32 tensor [0, 1], shape (3, H, W)
    # Ref: src/transforms.py:152 (ValTransform.__call__)
    t = TF.to_tensor(image)

    # Step 4: Normalize with mean/std from checkpoint (ImageNet stats)
    # Ref: src/transforms.py:153 (ValTransform.__call__)
    t = TF.normalize(t, mean=norm_mean, std=norm_std)

    return t


@app.post("/invocations")
async def invoke(request: Request):
    """
    Run Small Animal Classifier inference on a full camera-trap image.

    This model operates on the full frame (no bbox crop). The preprocessing
    pipeline strips the camera info-banner and squashes to a square before
    running the timm model.

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

        # Preprocess: banner crop → square resize → tensor → normalize
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # (3, H, W) → (1, 3, H, W)

        # Forward pass.
        # logits.float() ensures fp32 softmax regardless of input dtype.
        # Ref: src/run_inference.py:248-249 (infer)
        with torch.no_grad():
            logits = model(input_batch)
        probs = torch.softmax(logits.float(), dim=1).cpu()

        # Format as {class_name: confidence} for all classes
        scores = probs[0].tolist()
        predictions = {class_names[i]: scores[i] for i in range(len(scores))}

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
