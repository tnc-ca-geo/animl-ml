import io
import re
import json
from requests_toolbelt.multipart import decoder
from urllib.request import urlopen
import base64
from PIL import Image
from log_config import logger
import api_config
import boto3


PATTERN = re.compile('(?<=form-data; name=").*?(?=")')

client = boto3.client("runtime.sagemaker")


def create_response(status_code, message, headers=api_config.HEADERS):
    return {
        "statusCode": status_code,
        "headers": headers,
        "body": json.dumps(message)
    }


def handle_error(status_code, message):
    logger.debug("Error: {}".format(message))
    return create_response(status_code, {"message": message})


def crop(img, bbox):
    """
    API expects bbox as [ymin, xmin, ymax, xmax], in relative values,
    convert to tuple (xmin, ymin, xmax, ymax), in pixel values 
    """
    W, H = img.size
    if bbox:
        boxpx = (int(bbox[1]*W), int(bbox[0]*H), int(bbox[3]*W), int(bbox[2]*H))
        img = img.crop(boxpx)
    return img


def run_inference(img, bbox, models=api_config.MODELS):
    """
    Get class predictions from MIRA models
    """

    img = Image.open(img)
    img = crop(img, bbox)

    # convert to bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    output = {}
    for model in models: # TODO: invoke async
        name = model["endpoint_name"]
        output[name] = model
        predictions = {}
        try:
            response = client.invoke_endpoint(
                EndpointName = name,
                ContentType = "application/x-image", 
                Body = buf.getvalue()
            )

            # parse response
            response_body = response["Body"].read()
            response_body = response_body.decode("utf-8")
            pred = json.loads(response_body)["predictions"][0]
            for i in range(len(pred)):
                predictions[model["classes"][i]] = float(pred[i])
        except Exception as e:
            logger.debug("Error invoking {}: {}".format(name, e))
        output[name]["predictions"] = predictions

    return output


def parse_multipart_req(body, content_type):
    """
    Parse multipart-encoded form data
    """
    req = {}

    # convert to bytes if needed
    if type(body) is str:
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
    Event body must contain the following fields:

        image: the image to classify, encoded in base64, OR...
        url: url of remotely-hosted image    
        bbox (optional): bounding box of detected animal, represented as
            [ymin, xmin, ymax, xmax], in relative values
            
    Returns:
        A json object with MIRA model predictions
    """
    logger.debug("event: {}".format(event))

    try:
        event["body"] = base64.b64decode(event["body"])
    except Exception as e:
        msg = "Error decodeing image " + str(e)
        return handle_error(400, msg)

    headers = event.get("headers", {"Content-Type": ""})
    content_type = headers.get("Content-Type")

    if "multipart/form-data" in content_type:
        req = parse_multipart_req(event["body"], content_type)
        bbox = req.get("bbox")
        img = req.get("image") or req.get("url")
        if img:
            try:
                res = create_response(200, run_inference(img, bbox))
            except Exception as e:
                msg = "Error performing classification: " + str(e)
                res = handle_error(500, msg)
        else:
            msg = "No image or image URL present in form-data"
            res = handle_error(400, msg)
    else:   
        msg = "Content type is not multipart/form-data"
        res = handle_error(400, msg)

    return res