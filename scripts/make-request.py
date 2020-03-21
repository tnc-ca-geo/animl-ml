# Run inference on a localy hosted enpont
# The first argument must be a key for an image 
# already in the animl-images s3 bucket
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'CameraTraps/visualization')))
sys.path.append(os.path.abspath(os.path.join('..', 'CameraTraps/')))

print(sys.path)
import base64
import requests
import json
import argparse
from PIL import Image
import numpy as np
from visualization_utils import render_detection_bounding_boxes
import boto3

parser = argparse.ArgumentParser()
parser.add_argument("img_uri")
args = parser.parse_args()  

RENDER_THRESHOLD = 0.8
MODEL_NAME = 'saved_model_megadetector_v3_tf19'

if __name__ == "__main__":
    print("Detecting objects in: ", args.img_uri)

    # Send request
    image_paths = [args.img_uri]
    json_response = requests.post(
      "http://localhost:8080/invocations",
      data=image_paths[0],
      headers={
        "content-type":"application/json",
        "X-Amzn-SageMaker-Custom-Attribute":"tfs-model-name={}".format(MODEL_NAME)
      }
    )

    # Read predictions
    # print(json.loads(json_response.text))
    predictions = json.loads(json_response.text)['predictions']

    results = []
    for i, image_path in enumerate(image_paths):
        detections = []
        for box, clss, score in zip(predictions[i]['detection_boxes'], 
                                    predictions[i]['detection_classes'], 
                                    predictions[i]['detection_scores']):
            if score >= RENDER_THRESHOLD:
                detections.append({
                    'category': str(int(clss)),
                    'conf': score,
                    'bbox': [box[1], box[0], box[3] - box[1], box[2] - box[0]]
                })
        results.append({
            'file': image_path,
            'detections': detections
        })
    
    # Display results
    for res in results:
        print('result: ', res)
        # # im = Image.open(res['file'])

        # vis_utils.render_detection_bounding_boxes(
        #     res['detections'], 
        #     im, 
        #     confidence_threshold=RENDER_THRESHOLD)
        # im.save('output.jpg')