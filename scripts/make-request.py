import sys
import os
import argparse
import requests
import json
from PIL import Image
import numpy as np
sys.path.append(os.path.abspath(os.path.join("..", "CameraTraps/")))
# sys.path.append(os.path.abspath(os.path.join("..", "CameraTraps/visualization")))
from visualization.visualization_utils import render_detection_bounding_boxes

parser = argparse.ArgumentParser()
parser.add_argument("img_uri")
args = parser.parse_args()
URL = "http://localhost:8080/invocations"
RENDER_THRESHOLD = 0.8
MODEL_NAME = "megadetector"

if __name__ == "__main__":
    print("Detecting objects in: ", args.img_uri)

    headers = {
        "content-type": "application/x-image",
        "X-Amzn-SageMaker-Custom-Attribute": "tfs-model-name={}".format(MODEL_NAME)
    }

    # Open image in binary format
    with open(args.img_uri, "rb") as fd:

        # Post request
        r = requests.post(URL, data=fd, headers=headers)

        # Read predictions
        print("response: ", r.status_code)
        print(json.loads(r.text))
        predictions = json.loads(r.text)["predictions"]

        # results = []
        # for i, image_path in enumerate(image_paths):
        #     detections = []
        #     for box, clss, score in zip(predictions[i]["detection_boxes"], 
        #                                 predictions[i]["detection_classes"], 
        #                                 predictions[i]["detection_scores"]):
        #         if score >= RENDER_THRESHOLD:
        #             detections.append({
        #                 "category": str(int(clss)),
        #                 "conf": score,
        #                 "bbox": [box[1], box[0], box[3] - box[1], box[2] - box[0]]
        #             })
        #     results.append({
        #         "file": image_path,
        #         "detections": detections
        #     })
        
        # # Display results
        # for res in results:
        #     print("result: ", res)
        #     # Download image and render bounding box on it
        #     s3 = boto3.client("s3")
        #     with open("output/annotated_img.jpg", "wb") as f:
        #         s3.download_fileobj(BUCKET, args.img_uri, f)
        #         img=Image.open("output/annotated_img.jpg")
        #         render_detection_bounding_boxes(
        #             res["detections"], 
        #             img, 
        #             confidence_threshold=RENDER_THRESHOLD)
        #         img.save("output/annotated_img.jpg")