import io
import base64
import os
import time
import logging
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
import uvicorn
from megadetector.detection.pytorch_detector import PTDetector
from megadetector.detection.run_detector import load_detector


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Init config
detection_threshold = os.getenv("DETECTION_THRESHOLD", 0.0005)
model_path = os.getenv(
    "MODEL_PATH",
    "/opt/ml/model/md_v1000.0.0-redwood.pt",
)


# Load model
try:
    start_time = time.time()
    detection_threshold = float(detection_threshold)

    logger.info(f"Loading model from {model_path}")

    model = load_detector(model_path)
    if not isinstance(model, PTDetector):
        raise ValueError(
            "Megadetector v1000 is a PTDetector.  Ensure model file is a .pt file"
        )

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

    If the model load fails or the threshold is invalid,
    the container will have already crashed
    """
    return {"status": "Healthy"}


@app.post("/invocations")
async def invoke(request: Request):
    """
    Run MegaDetector inference on an image.

    The PTDetector handles preprocessing and NMS internally:
    - Preprocessing: https://github.com/agentmorris/MegaDetector/blob/main/megadetector/detection/pytorch_detector.py#L1177-L1183
    - NMS: https://github.com/agentmorris/MegaDetector/blob/main/megadetector/detection/pytorch_detector.py#L1305-L1314
    - Label mapping: https://github.com/agentmorris/MegaDetector/blob/main/megadetector/detection/run_detector.py#L63
    """
    request_start = time.time()
    try:
        # These are checked and gracefully handled in the loading step
        # This is just type assertions to help the lsp / editor
        assert isinstance(model, PTDetector)
        assert isinstance(detection_threshold, float)

        # Decode image
        decode_start = time.time()
        body = await request.body()
        try:
            image_bytes = base64.b64decode(body, validate=True)
        except Exception:
            image_bytes = body
        image = Image.open(io.BytesIO(image_bytes))
        decode_time = time.time() - decode_start
        logger.info(f"Image decoded in {decode_time:.3f}s - size: {image.size}")

        # Run inference
        inference_start = time.time()
        result = model.generate_detections_one_image(
            image, detection_threshold=detection_threshold
        )
        inference_time = time.time() - inference_start

        if result is None or result.get("failure") is not None:
            raise ValueError("Megadetector returned None")

        # Format results
        format_start = time.time()
        formatted_detections = []
        for det in result["detections"]:
            x, y, w, h = det["bbox"]
            formatted_detections.append(
                {
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "confidence": det["conf"],
                    "class": int(det["category"]),
                }
            )
        format_time = time.time() - format_start

        total_time = time.time() - request_start
        logger.info(
            f"Request completed in {total_time:.3f}s "
            f"(decode: {decode_time:.3f}s, inference: {inference_time:.3f}s, format: {format_time:.3f}s) "
            f"- {len(formatted_detections)} detections"
        )

        return formatted_detections

    except Exception as e:
        total_time = time.time() - request_start
        logger.error(f"Request failed after {total_time:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run inference: {e}")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
