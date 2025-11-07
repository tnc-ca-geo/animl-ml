"""Custom TorchServe model handler for camera-trap-vehicle-classifier model.
"""
from ts.torch_handler.image_classifier import ImageClassifier
import numpy as np
import base64
import json
import torch
from torchvision import transforms
import io
from PIL import Image, ImageOps
from ast import literal_eval

# mean/std values from running
# get_transforms_from_checkpoint() in run_vehicle_classifier.py
# https://github.com/agentmorris/camera-trap-vehicle-classifier/blob/main/run_vehicle_classifier.py
MEANS = np.asarray([0.48145466, 0.4578275, 0.40821073])
STDS = np.asarray([0.26862954, 0.26130258, 0.27577711])

# image size
IMG_SIZE = 448

class CustomImageClassifier(ImageClassifier):

    def initialize(self, context):
        super().initialize(context)
        # Set topk to the number of classes or less
        # NOTE: this is necessary for serving classifiers that have < 5 classes
        self.topk = min(4, getattr(self, "topk", 5))

    # define the transforms
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
        # custom pre-process code goes here
        """The preprocess function of MNIST program converts the input data to a float tensor
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            body = row.get("data") or row.get("body")
            body = json.loads(body)
            image = body.get("image")
            bbox = [0,0,1,1]
            if body.get("bbox"):
                print(f"bbox type: {type(body.get('bbox'))}")
                bbox = body.get("bbox")
                if isinstance(bbox, str):
                    bbox = literal_eval(body.get("bbox"))

            # print(f"image: {image}")
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

# adapted from: 
# https://github.com/microsoft/CameraTraps/blob/main/classification/crop_detections.py
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

    # if box_w == 0 or box_h == 0:
    #     tqdm.write(f'Skipping size-0 crop (w={box_w}, h={box_h}) at {save}')
    #     return False

    # Image.crop() takes box=[left, upper, right, lower]
    crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])

    if (box_w != box_h):
        # pad to square using 0s
        crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)

    print(f"cropped image size: {crop.size}")

    return crop
