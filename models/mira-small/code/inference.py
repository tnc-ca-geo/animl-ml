"""
Pre/Post-Processing for images submitted to MIRA-small endpoint for inference
The Nature Conservancy of California
"""

import logging
import json
from io import BytesIO
import numpy as np
from PIL import Image


log = logging.getLogger(__name__)

INPUT_SHAPE = (128, 128)
INSIZE = (1, 1, 128, 128)


def is_img_gray(image):
    """
    determine if the image is color or BW
    this is done by checking pixels in a line in the middle of the image
    if the RBG channels are the same, then it is grayscale
    """
    is_gray = True
    for i in range(image.size[0]):
        c = image.getpixel((i,image.size[1] / 2))
        c = np.asarray(c)
        mean = np.mean(c)
        tmp = True
        for b in c: tmp = tmp and (b == mean)
        if not tmp:
            is_gray = False
            break

    return is_gray


def input_handler(data, context):
    """
    Pre-process request input before it is sent to TensorFlow Serving REST API
    """
    log.info(" preprocessing request...")

    if context.request_content_type == "application/x-image":
        payload = data.read()
        image = Image.open(BytesIO(payload))

        # colour images will be converted to grayscale for now
        if not is_img_gray(image):
            image = image.convert('L')

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
        inp = json.dumps({
            "signature_name": "serving_default",
            "instances": img_data.tolist()
        })

        return inp

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def output_handler(data, context):
    """
    Post-process TensorFlow Serving output before it is returned to the client
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode("utf-8"))

    response_content_type = context.accept_header
    prediction = data.content

    return prediction, response_content_type
