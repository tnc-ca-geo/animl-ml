import logging
import json
from collections import namedtuple
import os
from io import BytesIO
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)
Context = namedtuple("Context",
                     "model_name, model_version, method, rest_uri, grpc_uri, "
                     "custom_attributes, request_content_type, accept_header")
INPUT_SHAPE = (128, 128)
INSIZE = (1, 1, 128, 128)

def input_handler(data, context):
    """ 
    Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request image (cropped), in format bytes object
        context (Context): an object containing request and configuration details

    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """

    log.info(" preprocessing request...")
    if context.request_content_type == "application/x-image":
        payload = data.read()
        log.info(" request recieved: {}".format(payload))
        image = Image.open(BytesIO(payload))
        
        # determine if the image is color or BW
        # this is done by checking pixels in a line in the middle of the image
        # if the RBG channels are the same, then it is grayscale
        isGray = True
        for i in range(image.size[0]):
            c = image.getpixel((i,image.size[1] / 2))
            c = np.asarray(c)
            mean = np.mean(c)
            tmp = True
            for b in c: tmp = tmp and (b == mean)
            if not tmp:
                isGray = False
                break

        # colour images will be converted to grayscale for now
        if not isGray: image = image.convert('L')

        image = image.resize(INPUT_SHAPE, Image.LANCZOS)

        # this procedure has to be the same as the one in datagen used for training
        # the pixel data is numerically shifted around its mean and scaled by its stdev
        image = np.asarray(image).astype(np.float32)
        image /= 255
        if len(image.shape) == 3: image = image[:,:,0]

        image -= np.mean(image)
        stdv = np.std(image)
        if stdv == 0: stdv = 1
        image /= stdv

        img_data = np.zeros(INSIZE, dtype=np.float32)
        img_data[0,0,] = image
    
        log.info(" binary -> np array conversion successful")
        log.info(" image shape: {}".format(img_data.shape))
        log.info(" making prediction...")
        inp = json.dumps({
            "signature_name": "serving_default",
            "instances": img_data.tolist()
        })
        return inp

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.

    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details

    Returns:
        (bytes, string): data to return to client, response content type
    """
    log.info(" output_handler firing")
    if data.status_code != 200:
        raise ValueError(data.content.decode("utf-8"))
    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type