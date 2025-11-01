"""Custom TorchServe model handler for NZDOC classifier.
"""
from ts.torch_handler.image_classifier import ImageClassifier
import numpy as np
# import cv2
import base64
import json
import torch
# from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
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

# mean/std values
MEANS = np.asarray([0.485, 0.456, 0.406])
STDS = np.asarray([0.229, 0.224, 0.225])

IMG_SIZE = 480
BUFFER = 0.1  # Extra space around the cropped image to ensure no animal parts missing (try playing with this)

class CustomImageClassifier(ImageClassifier):
    
    # define the transforms

    # NOTE: NZ DOC use the Ablumentations instead of torchvision for 
    # augmentation, which might produce different results. Compare both? 
    # they also normalize before converting to tensor. Not sure that matters:
    # A.Compose([A.Normalize(INPUT_MEAN, INPUT_STD), ToTensorV2()])
    image_processing = A.Compose([
        A.Normalize(MEANS, STDS),
        ToTensorV2()
    ])

    # image_processing = transforms.Compose([
    #     # resizes smaller edge to IMG_SIZE
    #     # NOTE: in NZDOC's cropping script, they use Pil to resize and
    #     # use resample=3. I'm unsure what resampling filter 3 refers to
    #     # as the Pillow docs are unclear
    #     transforms.Resize(IMG_SIZE, interpolation=Image.BICUBIC),
    #     transforms.CenterCrop(IMG_SIZE),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=MEANS, std=STDS, inplace=True),
    # ])

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
                image = crop_and_resize(image, bbox)
                augmented = self.image_processing(image=np.array(image))
                image = augmented['image']
                print(f"tensor shape fully processed: {image.shape}")

            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

def crop_and_resize(img, bbox_rel, out_size=IMG_SIZE, image_buffer=BUFFER):
    """
    Crops an image to the bounding box, and resizes to a square.

    NOTE: this replicates the behavior of NZ DOC's Crop_Resize_Image.py script, 
    in which they they crop to the dimensions of the bounding box and resize the
    arbitrary image shape to a square, thus distorting aspect ratio.

    It might be worth consulting with their ML engineers to ask whether they 
    think MSFT's approach (cropping to smallest square that encloses the bbox,
    and then resizing to preserve aspect ratio) might be better? Might also
    require retraining the model

    Args:
        img: PIL.Image.Image object, already loaded
        bbox_rel: list or tuple of float, [ymin, xmin, ymax, xmax] all in
            relative coordinates

    Returns: cropped image
    """

    print(f"cropping image. original image size: {img.size}")
    print(f"bbox_rel: {bbox_rel}")

    img_w, img_h = img.size
    left = bbox_rel[1]
    top = bbox_rel[0]
    right = bbox_rel[3]
    bottom = bbox_rel[2]

    print(f"left: {left}, top: {top}, right: {right}, bottom: {bottom}")

    # add buffer
    left = left - image_buffer
    if left < 0:
        left = 0
    top = top - image_buffer
    if top < 0:
        top = 0
    right = right + image_buffer
    if right > 1:
        right = 1
    bottom = bottom + image_buffer
    if bottom > 1:
        bottom = 1

    # Image.crop() takes box=[left, upper, right, lower]
    img = img.crop((
        int(left * img_w),
        int(top * img_h),
        int(right * img_w),
        int(bottom * img_h)
    ))
    img = img.resize((out_size, out_size), resample=3)
    print(f"cropped & resized image size: {img.size}")

    return img