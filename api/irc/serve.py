# FastAPI server for SpeciesNet model inference

import json
import logging
import os
import requests
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import base64
from PIL import Image, ImageOps
import io
from ast import literal_eval
from itertools import islice

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
  
# Initialize FastAPI app
app = FastAPI()

TF_SERVING_URL = "http://localhost:8501/v1/models/irc:predict"
CLASS_MAP_PATH = '/opt/ml/code/index_to_name.json'
INPUT_SIZE = 300 # model input size

@app.get("/ping")
async def ping():
    """Health check endpoint required by SageMaker"""
    return JSONResponse(content={"status": "healthy"}, status_code=200)

@app.post("/invocations")
async def invoke(request: Request):
    """SageMaker invocation endpoint with extended options"""
    try:
        logger.info(f"Content-Type: {request.headers.get('content-type')}")

        input_data = await request.json()

        # Validate required fields
        if 'image' not in input_data:
            return JSONResponse(
                status_code=400,
                content={"error": "Input must contain 'image' field"}
            )

        # Get bbox from request or use default [x_min, y_min, width, height]
        bbox = input_data.get('bbox', [0, 0, 1, 1])
        logger.info(f"bbox type: {type(bbox)}")
        if isinstance(bbox, str):
            bbox = literal_eval(input_data.get("bbox"))

        logger.info(f"bbox: {bbox}")
        image = input_data.get('image')
        logger.info(f"image type: {type(image)}")

        if isinstance(image, str):
            # if the image is a string of bytesarray.
            logger.info("Decoding base64 image string...")
            image = base64.b64decode(image)

        # If the image is sent as bytesarray
        if isinstance(image, (bytearray, bytes)):
            logger.info("Opening image from bytes...")
            image = Image.open(io.BytesIO(image))

            # always save as RGB for consistency
            if image.mode != 'RGB':
                image = image.convert(mode='RGB')
            
            # crop, resize
            image = crop(image, bbox)
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            logger.info(f"image size after resize: {image.size}")
      
        image_array = np.array(image)  # shape: (H, W, 3)
        logger.info(f"image array shape: {image_array.shape}")
        instances_dict = {
            "instances": [image_array.tolist()]
        }

        try:
            class_map = loadClassMap(CLASS_MAP_PATH)
            
            # Forward to TensorFlow Serving
            logger.info("Sending request to TensorFlow Serving...")
            tf_response = requests.post(TF_SERVING_URL, json=instances_dict)
            logger.info(f"TensorFlow Serving response status: {tf_response.status_code}")
            logger.info(f"TensorFlow Serving response content: {tf_response.text}")
            tf_response.raise_for_status()  # This will raise an HTTPError for 4xx/5xx
            pred = tf_response.json().get('predictions', [0])[0]

            classifications = {}
            for i in range(len(pred)):
                classifications[class_map[str(i)]] = float(pred[i])

            # sort and return top five classifications
            top_five = dict(sorted(classifications.items(), key=lambda item: item[1], reverse=True)[:5])
            logger.info(f"top five classifications: {top_five}")

            if tf_response is None:
                raise Exception("No predictions returned from model")

            return JSONResponse(content=top_five, status_code=tf_response.status_code)
        except Exception as e:
            logger.error(f"Error communicating with TensorFlow Serving: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Model error: {str(e)}"}
            )

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# adapted from: 
# https://github.com/microsoft/CameraTraps/blob/main/classification/crop_detections.py
def crop(img, bbox_rel):
    """
    Crops an image to the tightest square enclosing each bounding box. 
    This will always generate a square crop whose size is the larger of the 
    bounding box width or height. In the case that the square crop boundaries 
    exceed the original image size, the crop is padded with 0s.

    Args:
        img: PIL.Image.Image object, already loaded
        bbox_rel: list or tuple of float, [ymin, xmin, ymax, xmax] all in
            relative coordinates

    Returns: cropped image
    """

    print(f"cropping image. original image size: {img.size}")

    img_w, img_h = img.size
    xmin = int(bbox_rel[1] * img_w)
    ymin = int(bbox_rel[0] * img_h)
    box_w = int((bbox_rel[3] - bbox_rel[1]) * img_w)
    box_h = int((bbox_rel[2] - bbox_rel[0]) * img_h)

    # expand box width or height to be square, but limit to img size
    box_size = max(box_w, box_h)
    xmin = max(0, min(
        xmin - int((box_size - box_w) / 2),
        img_w - box_w))
    ymin = max(0, min(
        ymin - int((box_size - box_h) / 2),
        img_h - box_h))
    box_w = min(img_w, box_size)
    box_h = min(img_h, box_size)

    # if box_w == 0 or box_h == 0:
    #     tqdm.write(f'Skipping size-0 crop (w={box_w}, h={box_h}) at {save}')
    #     return False

    # Image.crop() takes box=[left, upper, right, lower]
    crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])

    if (box_w != box_h):
        # pad to square using 0s
        crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)

    print(f"cropped image size: {crop.size}")

    return crop

def loadClassMap(class_map_path):
    """Load class map from JSON file"""
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)
    return class_map

def main():
    """Run the server"""
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    main()
