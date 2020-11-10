import logging
import json
from collections import namedtuple
import os
from io import BytesIO
import boto3
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)
Context = namedtuple("Context",
                     "model_name, model_version, method, rest_uri, grpc_uri, "
                     "custom_attributes, request_content_type, accept_header")

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request data, in format bytes object
        context (Context): an object containing request and configuration details

    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """

    log.info(" preprocessing request...")
    if context.request_content_type == "application/x-image":
        payload = data.read()
        log.info(" request recieved: {}".format(payload))
        image = Image.open(BytesIO(payload))
        if image.mode not in ("RGBA", "RGB", "L"):
            raise AttributeError(f"Image uses unsupported mode {image.mode}")
        if image.mode == "RGBA" or image.mode == "L":
            image = image.convert(mode="RGB")
        np_img=np.asarray(image, np.uint8)
        np_img_expanded = np.expand_dims(np_img, axis=0)
        log.info(" binary -> np array conversion successful")
        log.info(" image shape: {}".format(np_img_expanded.shape))
        log.info(" making prediction...")
        inp = json.dumps({
            "signature_name": "serving_default",
            "instances": np_img_expanded.tolist()
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