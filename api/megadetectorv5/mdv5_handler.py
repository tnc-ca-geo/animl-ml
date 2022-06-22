from ts.torch_handler.object_detector import ObjectDetector
import torch
import numpy as np
import base64
from PIL import Image
from io import BytesIO

class Yolov5TorchscriptModelHandler(ObjectDetector):
    """
    A custom model handler implementation.
    """

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """

        def three_channel_arr_to_shape(a, shape):
            pad_y_left, pad_y_right, pad_x_left, pad_x_right = get_pad_lengths(a,shape)
            return np.pad(a,((pad_y_left, pad_y_right), 
                            (pad_x_left, pad_x_right), (0,0)),
                        mode = 'constant')

        def get_pad_lengths(a, shape):
            y_, x_ = shape
            y, x, c = a.shape
            y_pad = (y_-y)
            x_pad = (x_-x)
            pad_y_left = y_pad//2
            pad_y_right = y_pad//2 + y_pad%2
            pad_x_left = x_pad//2
            pad_x_right = x_pad//2 + x_pad%2
            return pad_y_left, pad_y_right, pad_x_left, pad_x_right

        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")

        ### below are custom steps for yolov5
        # Take the input data and make it inference ready
        print(type(preprocessed_data))
        if isinstance(preprocessed_data, str):
            # if the image is a string of bytesarray.
            preprocessed_data = base64.b64decode(preprocessed_data)
        print(type(preprocessed_data))
        # If the image is sent as bytesarray
        if isinstance(preprocessed_data, (bytearray, bytes)):
            preprocessed_data = np.array(Image.open(BytesIO(preprocessed_data)))
        else:
            raise ValueError("The POST request should contain a bytearray. Or string of bytesarray.")
        print(type(preprocessed_data))
        print(preprocessed_data.shape)
        preprocessed_data = three_channel_arr_to_shape(preprocessed_data, (2048,2048))
        print(preprocessed_data.shape)
        preprocessed_data = torch.from_numpy(preprocessed_data)
        print(preprocessed_data.shape)
        preprocessed_data = torch.moveaxis(preprocessed_data,2,0).float()[None,...]
        print(preprocessed_data.shape)

        return preprocessed_data

    # def postprocess(self, inference_output):
    #     """
    #     Return inference result.
    #     :param inference_output: list of inference output
    #     :return: list of predict results
    #     """
    #     # Take output from network and post-process to desired format
    #     postprocess_output = inference_output
    #     return postprocess_output
