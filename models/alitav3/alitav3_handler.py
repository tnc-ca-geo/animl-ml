"""Custom TorchServe model handler for Alita v3 classifier model.

Based on the original Alita inference code:
https://github.com/Wologman/Alita/blob/main/inference.py
"""
from ts.torch_handler.image_classifier import ImageClassifier
import numpy as np
import base64
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
import io
from PIL import Image, ImageOps
from ast import literal_eval

# Mean/std values from original Alita inference code
# https://github.com/Wologman/Alita/blob/e0e26ab76908cede90dc403f55d5ce189e44af57/inference.py#L108C1-L109C107
MEANS = np.asarray([0.485, 0.456, 0.406])
STDS = np.asarray([0.229, 0.224, 0.225])

# Image size for Alita v3
IMG_SIZE = 480

class AlitaV3Handler(ImageClassifier):

    def initialize(self, context):
        super().initialize(context)
        # Set topk to the number of classes or less
        # Alita v3 has 79 classes, so we'll use all of them
        self.topk = min(79, getattr(self, "topk", 5))

    # Define the transforms matching original Alita preprocessing
    image_processing = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEANS, std=STDS)
    ])

    def preprocess(self, data):
        """
        Overriding this method for custom preprocessing.
        :param data: raw data to be transformed
        :return: preprocessed data for model input
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            body = row.get("data") or row.get("body")
            body = json.loads(body)
            image = body.get("image")
            bbox = [0, 0, 1, 1]
            if body.get("bbox"):
                print(f"bbox type: {type(body.get('bbox'))}")
                bbox = body.get("bbox")
                if isinstance(bbox, str):
                    bbox = literal_eval(body.get("bbox"))

            print(f"bbox: {bbox}")
            print(f"image type: {type(image)}")
            print(f"bbox type: {type(bbox)}")

            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))

                # always save as RGB for consistency
                if image.mode != 'RGB':
                    image = image.convert(mode='RGB')
                
                # crop, resize, and convert to tensor
                image = crop(image, bbox)
                image = self.image_processing(image)
                print(f"tensor shape fully processed: {image.shape}")

            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    def postprocess(self, data):
        """
        Apply sigmoid activation to match original Alita inference behavior.
        Based on: https://github.com/Wologman/Alita/blob/e0e26ab76908cede90dc403f55d5ce189e44af57/inference.py#L712C1-L713C42
        """
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(data)
        
        # Get top-k predictions
        ps, classes = torch.topk(probabilities, self.topk, dim=1)
        
        # Convert to list format expected by TorchServe
        results = []
        for i in range(data.shape[0]):
            result = {}
            for j in range(self.topk):
                class_idx = classes[i][j].item()
                prob = ps[i][j].item()
                class_name = self.mapping.get(str(class_idx), str(class_idx))
                result[class_name] = prob
            results.append(result)
        
        return results


def crop(img, bbox_rel):
    """
    Crops an image to the tightest square enclosing each bounding box. 
    This will always generate a square crop whose size is the larger of the 
    bounding box width or height. In the case that the square crop boundaries 
    exceed the original image size, the crop is padded with 0s.

    Args:
        img: PIL.Image.Image object, already loaded
        bbox_rel: list or tuple of float, [ymin, xmin, ymax, xmax] all in
            relative coordinates

    Returns: cropped image
    """
    print(f"cropping image. original image size: {img.size}")

    img_w, img_h = img.size
    xmin = int(bbox_rel[1] * img_w)
    ymin = int(bbox_rel[0] * img_h)
    box_w = int((bbox_rel[3] - bbox_rel[1]) * img_w)
    box_h = int((bbox_rel[2] - bbox_rel[0]) * img_h)

    # expand box width or height to be square, but limit to img size
    box_size = max(box_w, box_h)
    xmin = max(0, min(
        xmin - int((box_size - box_w) / 2),
        img_w - box_w))
    ymin = max(0, min(
        ymin - int((box_size - box_h) / 2),
        img_h - box_h))
    box_w = min(img_w, box_size)
    box_h = min(img_h, box_size)

    # Image.crop() takes box=[left, upper, right, lower]
    crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])

    if (box_w != box_h):
        # pad to square using 0s
        crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)

    print(f"cropped image size: {crop.size}")

    return crop