"""
HWI-ADS-v1 (Hawaiian Wildlife Invasives) classification model server.

This model uses a SpeciesNet EfficientNet V2 M backbone with a fine-tuned
linear classification head, trained by Addax Data Science for USDA Forest
Service (IPIF) & The Nature Conservancy (TNC).

Architecture adapted from the AddaxAI FXClassifier framework:
https://github.com/PetervanLunteren/AddaxAI

Model weights from:
https://huggingface.co/Addax-Data-Science/HWI-ADS-v1
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
import torch.nn as nn
import torch.nn.functional as F
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
# Model Architecture
# =============================================================================

def load_fx_checkpoint(weights_path, map_location="cpu"):
    """Load the SpeciesNet backbone (an onnx2torch GraphModule)."""
    try:
        from torch.serialization import add_safe_globals
        from torch.fx.graph_module import reduce_graph_module
        add_safe_globals([reduce_graph_module])
    except Exception:
        pass

    try:
        obj = torch.load(weights_path, map_location=map_location, weights_only=True)
    except Exception:
        obj = torch.load(weights_path, map_location=map_location, weights_only=False)

    if hasattr(obj, "state_dict") and hasattr(obj, "forward"):
        return obj
    raise ValueError("Expected a torch.nn.Module GraphModule")


class FXClassifier(nn.Module):
    """SpeciesNet backbone + linear classification head."""

    def __init__(self, backbone, num_classes, in_features):
        super().__init__()
        self.backbone = backbone

        # Freeze backbone — we only trained the head
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """Image tensor (NCHW) → class logits."""
        # Convert NCHW → NHWC for the backbone (originates from TensorFlow/ONNX)
        x = x.permute(0, 2, 3, 1).contiguous()

        # Backbone outputs a flat 2D tensor [batch, 2498] — the onnx2torch
        # conversion includes global average pooling within the backbone itself
        z = self.backbone(x)
        z = z.flatten(1)

        return self.head(z)


# =============================================================================
# Load Model
# =============================================================================

backbone_path = os.getenv(
    "BACKBONE_PATH",
    "/opt/ml/model/always_crop_99710272_22x8_v12_epoch_00148.pt",
)
checkpoint_path = os.getenv(
    "CHECKPOINT_PATH",
    "/opt/ml/model/final-20260317.pt",
)

try:
    start_time = time.time()
    logger.info("Loading model...")

    # Load the checkpoint (contains head weights + metadata)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load the backbone (frozen SpeciesNet feature extractor)
    backbone = load_fx_checkpoint(backbone_path, map_location="cpu")

    # Read head input size from the checkpoint's trained weights
    in_features = checkpoint["model"]["head.weight"].shape[1]

    # Assemble the full model (backbone + head) and load trained weights
    model = FXClassifier(
        backbone=backbone,
        num_classes=checkpoint["num_classes"],
        in_features=in_features,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Read class names and build preprocessing pipeline
    class_names = checkpoint["class_names"]
    norm_params = checkpoint["normalize"]
    preprocess = transforms.Compose([
        transforms.Resize((checkpoint["img_size"], checkpoint["img_size"]), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"]),
    ])

    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f}s with {len(class_names)} classes")

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


def get_crop(image, bbox):
    """
    Crop detection from image using a simple tight crop.
    Matches the training-time cropping used by Addax Data Science.
    Ref: https://github.com/PetervanLunteren/AddaxAI/blob/main/classification_utils/model_types/addax-sppnet/classify_detections.py

    Args:
        image: PIL Image (full resolution)
        bbox: Normalized [x, y, width, height] in range [0.0, 1.0]

    Returns:
        Cropped PIL Image
    """
    W, H = image.size
    x, y, w, h = bbox

    left = max(0, int(round(x * W)))
    top = max(0, int(round(y * H)))
    right = min(W, int(round((x + w) * W)))
    bottom = min(H, int(round((y + h) * H)))

    if right <= left or bottom <= top:
        return image

    return image.crop((left, top, right, bottom))


@app.post("/invocations")
async def invoke(request: Request):
    """
    Run HWI-ADS-v1 classification on a cropped image.

    Args:
        request: JSON with 'image' (base64) and 'bbox' ([x, y, w, h] normalized)
    Returns:
        Dictionary of class names to confidence scores
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
        bbox = payload["bbox"]

        # Crop to the detection bounding box
        crop = get_crop(image, bbox)

        # Preprocess: resize to 480x480, convert to tensor, normalize
        input_tensor = preprocess(crop)
        input_batch = input_tensor.unsqueeze(0)  # [3,480,480] → [1,3,480,480]

        # Run inference: logits → softmax → probabilities
        with torch.no_grad():
            output = model(input_batch)
            probabilities = F.softmax(output, dim=1)

        # Format results as {class_name: confidence}
        scores = probabilities[0].tolist()
        predictions = {class_names[i]: scores[i] for i in range(len(scores))}

        total_time = time.time() - request_start
        logger.info(f"Request completed in {total_time:.3f}s - {len(predictions)} classes")

        return predictions

    except Exception as e:
        total_time = time.time() - request_start
        logger.error(f"Request failed after {total_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run inference: {e}")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
