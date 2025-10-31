"""
MIRA API - Species detection for camera trap images
The Nature Conservancy of California
"""

import io
import re
import json
import base64
from urllib.request import urlopen
from requests_toolbelt.multipart import decoder
from PIL import Image, ImageFile
import botocore
import boto3
from log_config import logger
import api_config

PATTERN = re.compile('(?<=form-data; name=").*?(?=")')
client = boto3.client("runtime.sagemaker")

def create_response(status_code, message, headers=api_config.HEADERS):
    """
    Create response to return to client
    """
    return {
        "statusCode": status_code,
        "headers": headers,
        "body": json.dumps(message)
    }


def handle_error(status_code, message):
    """
    Log error and generate response for client
    """
    logger.debug("Error: %s", message)
    return create_response(status_code, {"message": message})


def predict(model, img_data):
    """
    Get predictions from model endpoint
    """
    predictions = {}
    try:
        response = client.invoke_endpoint(
            EndpointName = model["endpoint_name"],
            ContentType = "application/x-image",
            Body = img_data
        )

        # parse response
        response_body = response["Body"].read()
        response_body = response_body.decode("utf-8")
        pred = json.loads(response_body)["predictions"][0]
        for i, _ in enumerate(pred):
            predictions[model["classes"][i]] = float(pred[i])
    except botocore.exceptions.ClientError as err:
        logger.debug("Error invoking MIRA model endpoint: %s", err)
    return predictions


def get_bbox(img, bbox):
    """
    API expects bbox as [ymin, xmin, ymax, xmax], in relative values,
    convert to tuple (xmin, ymin, xmax, ymax), in pixel values
    """
    width, height = img.size
    bbox_px = (0, 0, width, height)
    if bbox:
        bbox_px = (
          int(bbox[1] * width),
          int(bbox[0] * height),
          int(bbox[3] * width),
          int(bbox[2] * height)
        )
    return bbox_px


def run_inference(img, bbox, models=api_config.MODELS):
    """
    Get class predictions from MIRA models
    """
    output = {}
    img = Image.open(img)
    img = img.crop(get_bbox(img, bbox))

    # convert to bytes array
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    # TODO: call endpoints async
    for model in models:
        name = model["endpoint_name"]
        output[name] = model
        output[name]["predictions"] = predict(model, buf.getvalue())
    return output


def parse_multipart_req(body, content_type):
    """
    Parse multipart-encoded form data
    """
    req = {}

    # convert to bytes if needed
    if isinstance(body, str):
        body = bytes(body, "utf-8")

    multipart_data = decoder.MultipartDecoder(body, content_type)
    for part in multipart_data.parts:
        content_disposition = part.headers.get(b"Content-Disposition", b"")
        content_disposition = content_disposition.decode("utf-8")
        search_field = PATTERN.search(content_disposition)
        if search_field:
            if search_field.group(0) == "image":
                img_io = io.BytesIO(part.content)
                img_io.seek(0)
                req["image"] = img_io
            elif search_field.group(0) == "url":
                url = part.content.decode("utf-8")
                img_io = io.BytesIO(urlopen(url).read())
                req["url"] = img_io
            elif search_field.group(0) == "bbox":
                req["bbox"] = json.loads(part.content.decode("utf-8"))
            else:
                logger.debug("Bad field name in form-data")
    return req


def handler(event, context):
    """
    Handle MIRA classification requests

    Must be submitted as a POST request encoded as multipart/form-data
    Form must contain the following parts:
        - image: the image to classify, encoded in base64, OR...
        - url: url of remotely-hosted image
        - bbox (optional): bounding box of detected animal, represented as
            [ymin, xmin, ymax, xmax], in relative values

    Returns:
        A json object with MIRA model predictions
    """
    try:
        event["body"] = base64.b64decode(event["body"])
    except base64.binascii.Error as err:
        msg = "Error decodeing image " + str(err)
        return handle_error(400, msg)

    headers = event.get("headers", {"Content-Type": ""})
    content_type = headers.get("Content-Type") or headers.get("content-type")
    if "multipart/form-data" in content_type:
        req = parse_multipart_req(event["body"], content_type)
        bbox = req.get("bbox")
        img = req.get("image") or req.get("url")
        if img:
            try:
                res = create_response(200, run_inference(img, bbox))
            except Exception as err:
                msg = "Error performing classification: " + str(err)
                res = handle_error(500, msg)
        else:
            msg = "No image or image URL present in form-data"
            res = handle_error(400, msg)
    else:
        msg = "Content type is not multipart/form-data"
        res = handle_error(400, msg)

    return res
