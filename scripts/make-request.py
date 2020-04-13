# Run inference on a localy hosted enpont
# The first argument must be a key for an image 
# already in the animl-images s3 bucket
import sys
import os
import argparse
import requests
import json
from PIL import Image
import numpy as np
import boto3
sys.path.append(os.path.abspath(os.path.join('..', 'CameraTraps/')))
# sys.path.append(os.path.abspath(os.path.join('..', 'CameraTraps/visualization')))
from visualization.visualization_utils import render_detection_bounding_boxes

parser = argparse.ArgumentParser()
parser.add_argument("img_uri")
args = parser.parse_args()

BUCKET = "animl-images"
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
    print(json.loads(json_response.text))
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
        # Download image and render bounding box on it
        s3 = boto3.client('s3')
        with open('output/annotated_img.jpg', 'wb') as f:
            s3.download_fileobj(BUCKET, args.img_uri, f)
            img=Image.open('output/annotated_img.jpg')
            render_detection_bounding_boxes(
                res['detections'], 
                img, 
                confidence_threshold=RENDER_THRESHOLD)
            img.save('output/annotated_img.jpg')