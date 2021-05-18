"""
MIRA API test script
Submit an image & bounding box for species detection for camera trap images
The Nature Conservancy of California
"""

import os
import json
import argparse
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--img", help = "filename of local image to test")
parser.add_argument("--img-url", help = "url of remotely-hosted image to test")
parser.add_argument("--bbox", help = "bounding box of object in image")
parser.add_argument("--local",
                    help = "test api running locally",
                    action="store_true")
args = parser.parse_args()


LOCAL_URL = "http://localhost:3000/dev/classify"
DEV_URL = "https://9q9u0ns6ul.execute-api.us-west-2.amazonaws.com/dev/classify"
API_URL =  LOCAL_URL if args.local else DEV_URL
TEST_IMG_DIR = os.path.abspath(
  os.path.join(os.path.dirname(__file__), "..", "..", "..", "input"))

params = {
    # 'confidence': 0.8,
    # 'render': True
}


def handle_response(res):
    """
    Handle preditction response
    """
    print(res)
    print("response: {}".format(res.text))


def bbox_to_list(bbox_string):
    """
    Convert bounding box passed in as string such as:
    "[0.536007, 0.434649, 0.635773, 0.543599]" to list of floats
    """
    bbox = bbox_string.strip('][').split(', ')
    bbox = [float(x) for x in bbox]
    return bbox


def request_inference(img, img_url, bbox):
    """
    Prep and send image as mulitpart form-data
    """
    fields = {}
    if img_url:
        fields['url'] = img_url
    if img:
        img_path = os.path.join(TEST_IMG_DIR, img)
        fields['image'] = (img, open(img_path, 'rb'), 'image/jpeg')
    if bbox:
        bbox = bbox_to_list(bbox)
        fields['bbox'] = json.dumps(bbox)

    print("Posting inference request to {}".format(API_URL))
    print("with fields: {}".format(fields))
    multipart_data = MultipartEncoder(fields = fields)
    res = requests.post(API_URL,
                      params = params,
                      data = multipart_data,
                      headers = {'Content-Type': multipart_data.content_type})
    return res


if __name__ == "__main__":
    if args.img or args.img_url:  
        r = request_inference(args.img, args.img_url, args.bbox)
        handle_response(r)
    else:
        print("Supply either an image filename or url to submit for inference")
        print("Run test_api.py --help for usage info")
