"""
Pre/Post-Processing for images submitted to IRC model served via TensorFlow Serving REST API
The Nature Conservancy of California
"""

import logging
import json
from io import BytesIO
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

INPUT_SHAPE = (300, 300)
INSIZE = (1, 300, 300, 1)  # batch, height, width, channels

def input_handler(data, context):
    """
    Pre-process request input before it is sent to TensorFlow Serving REST API
    """
    log.info(" preprocessing request...")

    if context.request_content_type == "application/x-image":
        payload = data.read()
        print(f"payload type: {type(payload)}")
        print(f"data: {data}")
        image = Image.open(BytesIO(payload))


        image = image.resize(INPUT_SHAPE, Image.LANCZOS)

        # this procedure has to be the same as the one in datagen used
        # for training. The pixel data is numerically shifted around its mean
        # and scaled by its stdev
        image = np.asarray(image).astype(np.float32)
        image /= 255
        if len(image.shape) == 3:
            image = image[:,:,0]

        image -= np.mean(image)
        stdv = np.std(image)
        if stdv == 0:
            stdv = 1
        image /= stdv

        img_data = np.zeros(INSIZE, dtype=np.float32)
        img_data[0,0,] = image

        # format input
        return json.dumps({
            "signature_name": "serving_default",
            "instances": img_data.tolist()
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def output_handler(data, context):
    """
    Post-process TensorFlow Serving output before it is returned to the client
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode("utf-8"))
    return data.content, context.accept_header


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
