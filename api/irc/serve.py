# FastAPI server for SpeciesNet model inference

import json
import logging
import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import base64
from PIL import Image, ImageOps
import io
from ast import literal_eval

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

TF_SERVING_URL = "http://localhost:8501/v1/models/irc:predict"

@app.get("/ping")
async def ping():
    """Health check endpoint required by SageMaker"""
    logger.info("ping received")
    return JSONResponse(content={"status": "healthy"}, status_code=200)

@app.post("/invocations")
async def invoke(request: Request):
    """SageMaker invocation endpoint with extended options"""
    logger.info("invoke received")
    try:
        logger.info("Reading request body...")
        logger.info(f"Content-Type: {request.headers.get('content-type')}")

        # Get raw request body
        # body = await request.body()
        # logger.info(f"Request body: {body.decode('utf-8')}")
        # input_data = json.loads(body.decode('utf-8'))
        input_data = await request.json()
        
        # print key input parameters
        logger.info('input_data keys: %s', input_data.keys())

        # Validate required fields
        if 'image' not in input_data:
            return JSONResponse(
                status_code=400,
                content={"error": "Input must contain 'image' field"}
            )
        
        # # Get  parameters with defaults
        # components = input_data.get('components', 'all')
        # if components not in ['all', 'classifier', 'detector']:
        #     return JSONResponse(
        #         status_code=400,
        #         content={"error": "components must be one of: all, classifier, detector"}
        #     )

        # geofence = input_data.get('geofence', True)
        # batch_size = input_data.get('batch_size', 8)

        # # Validate batch_size is integer
        # if not isinstance(batch_size, int):
        #     return JSONResponse(
        #         status_code=400,
        #         content={"error": "batch_size must be an integer"}
        #     )

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
            
            # crop, resize, and convert to tensor
            image = crop(image, bbox)
            # image = self.image_processing(image)
            # print(f"tensor shape fully processed: {image.shape}")

        # Create temporary file for the image

        # Decode and save image
        # image_bytes = base64.b64decode(input_data['image'])
        # image = Image.open(io.BytesIO(image_bytes))
        temp_path = "/tmp/temp_image.jpg"
        image.save(temp_path)

        # Create instances dict
        instances_dict = {
            "instances": [{
                "filepath": temp_path
            }]
        }

        # # Add  country parameter
        # # TODO: validate this is 3 letter ISO
        # if 'country' in input_data:
        #     instances_dict['instances'][0]['country'] = input_data['country']

        # # Add admin1_region parameter
        # if 'admin1_region' in input_data:
        #     instances_dict['instances'][0]['admin1_region'] = input_data['admin1_region']

        print('instances_dict', instances_dict)
        try:
            # # Update geofencing setting
            # model.geofence = geofence

            # # Run prediction based on components
            # if components == "classifier":
            #     # Get bbox from request or use default [x_min, y_min, width, height]
            #     bbox = input_data.get('bbox', [0, 0, 1, 1])

            #     # Create detections dict with bbox
            #     detections_dict = {
            #         temp_path: {
            #             "detections": [{
            #                 "bbox": bbox
            #             }]
            #         }
            #     }

            #     predictions_dict = model.classify(
            #         instances_dict=instances_dict,
            #         detections_dict=detections_dict,
            #         batch_size=batch_size
            #     )
            # elif components == "detector":
            #     predictions_dict = model.detect(
            #         instances_dict=instances_dict
            #     )
            # else:  # all components
            #     predictions_dict = model.predict(
            #         instances_dict=instances_dict,
            #         batch_size=batch_size
            #     )


            # Forward to TensorFlow Serving
            tf_response = requests.post(TF_SERVING_URL, json=instances_dict)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if tf_response is None:
                raise Exception("No predictions returned from model")

            return JSONResponse(content=tf_response.json(), status_code=tf_response.status_code)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return JSONResponse(
                status_code=500,
                content={"error": f"Model error: {str(e)}"}
            )

    except Exception as e:
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

def main():
    """Run the server"""
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    main()
