import argparse
import requests
import os
import json
from requests_toolbelt.multipart.encoder import MultipartEncoder

parser = argparse.ArgumentParser()
parser.add_argument("img_uri")
args = parser.parse_args()

API_URL = "https://2xuiw1fidh.execute-api.us-west-1.amazonaws.com/dev/classify"
TEST_IMG_DIR = os.path.abspath(
  os.path.join(os.path.dirname(__file__), "..", "..", "..", "input"))

params = {
    # 'confidence': 0.8,
    # 'render': True
}

files = {}

def handle_response(r):
    print("response: {}".format(r.text))

def request_inference(img_uri):
    if not args.img_uri.lower().endswith('.jpg'):
        return
    img_path = os.path.join(TEST_IMG_DIR, img_uri)

    multipart_data = MultipartEncoder(
        fields = {
            'image': (img_uri, open(img_path, 'rb'), 'image/jpeg'),
            # 'url': '[DOWNLOAD_URL]'
            'bbox': json.dumps([0.536007, 0.434649, 0.635773, 0.543599]),
        }
    )
    r = requests.post(API_URL,
                      params=params,
                      data=multipart_data,
                      headers={'Content-Type': multipart_data.content_type})    
    return r

if __name__ == "__main__":
    print("Detecting objects in: ", args.img_uri)
    r = request_inference(args.img_uri)
    handle_response(r)