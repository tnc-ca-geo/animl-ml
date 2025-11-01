"""Custom TorchServe model handler for SDZWA Southwest v3 model.
"""
from ts.torch_handler.image_classifier import ImageClassifier
import numpy as np
# import cv2
import base64
import json
import torch
from torchvision import transforms
# import torchvision
import io
from PIL import Image, ImageOps
# from io import BytesIO
# from typing import Union
# import os
# np.random.seed(42)
# torch.manual_seed(42)
# os.environ["PYTHONHASHSEED"] = "42"
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.set_num_threads(1)
from ast import literal_eval

# mean/std values from 
# https://github.com/microsoft/CameraTraps/blob/main/classification/train_classifier.py
MEANS = np.asarray([0.485, 0.456, 0.406])
STDS = np.asarray([0.229, 0.224, 0.225])

# image size
IMG_SIZE = 299

class CustomImageClassifier(ImageClassifier):
    
    # define the transforms
    image_processing = transforms.Compose([
        # torch.resize order is H,W
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # resizes smaller edge to IMG_SIZE
        # transforms.Resize(IMG_SIZE, interpolation=Image.BICUBIC),
        # transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        # transforms.Normalize(mean=MEANS, std=STDS, inplace=True),
    ])

    def preprocess(self, data):
        """
        Overriding this method for custom preprocessing.
        :param data: raw data to be transformed
        :return: preprocessed data for model input
        """
        # custom pre-procsess code goes here
        print(f"data: {data}")
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
