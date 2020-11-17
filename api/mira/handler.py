import io
import re
import json
from requests_toolbelt.multipart import decoder
from urllib.request import urlopen
import base64
from PIL import Image
from log_cfg import logger
import boto3

PATTERN = re.compile('(?<=form-data; name=").*?(?=")')
MODELS = [
  {
			"model": "large",
			"endpoint_name": "tensorflow-inference-2020-11-14-16-33-30-130",
			"classes": ["fox", "skunk", "empty"]
  },
  {
    "model": "small",
    "endpoint_name": "tensorflow-inference-2020-11-14-21-37-17-458",
    "classes": ["rodent", "empty"]
  }
]

client = boto3.client("runtime.sagemaker")

def run_inference(img, bbox, models=MODELS):
    """
    Get class predictions from MIRA models
    """

    # Megadetector bbox is [ymin, xmin, ymax, xmax] in relative values
    # convert to tuple (xmin, ymin, xmax, ymax) in pixel values 
    W, H = img.size
    if bbox:
        boxpx = (int(bbox[1]*W), int(bbox[0]*H), int(bbox[3]*W), int(bbox[2]*H))
        img = img.crop(boxpx)

    # convert to bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    output = {}
    for model in models:  # TODO: invoke async
        output[model["model"]] = model
        predictions = {}
        # invoke endpoint
        print("invoking mira-{}".format(model))
        response = client.invoke_endpoint(
            EndpointName=model["endpoint_name"],
            ContentType="application/x-image", 
            Body=buf.getvalue()
        )
        # parse response
        response_body = response["Body"].read()
        response_body = response_body.decode("utf-8")
        pred = json.loads(response_body)["predictions"][0]
        print("predictions: {}".format(pred))
        for i in range(len(pred)):
            predictions[model["classes"][i]] = float(pred[i])
        output[model["model"]]["predictions"] = predictions
    print("model output: {} ".format(output))
    return output

def parse_multipart_req(body, content_type):
    """
    Parse multipart-encoded form data
    """
    req = {}

    # convert to bytes if need
    if type(body) is str:
        body = bytes(body,"utf-8")

    multipart_data = decoder.MultipartDecoder(body, content_type)
    for part in multipart_data.parts:
        content_disposition = part.headers.get(b"Content-Disposition", b"").decode("utf-8")
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
            print("Bad field name in form-data")
    return req

def classify(event, context):
    logger.debug("event: {}".format(event))
    res = list()

    # validate request
    assert event.get("httpMethod") == "POST"
    try :
        event["body"] = base64.b64decode(event["body"])
    except :
         return {
            "statusCode": 400,
            "body": json.dumps(res)
        }

    # # check that the content uploaded is not too big
    # # request.content_length is the length of the total payload
    # # also will not proceed if cannot find content_length, hence in the else we exceed the max limit
    # content_length = event.content_length
    # print("content_length: {}".format(content_length))

    content_type = event.get("headers", {"Content-Type": ""}).get("Content-Type")
    if "multipart/form-data" in content_type:
        req = parse_multipart_req(event["body"], content_type)
        img = None
        if req["image"]:
            img = Image.open(req["image"])
        elif req["url"]:
            img = Image.open(req["url"])
        else:
            print("No image or image URL present in form-data")
        bbox = req["bbox"] if req["bbox"] else None
        res.append(run_inference(img, bbox))

    return {
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
            "statusCode": 200,
            "body": json.dumps(res)
            }
