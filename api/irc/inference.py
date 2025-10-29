# inference.py
# Custom inference script for AWS SageMaker TensorFlow Serving
# This script defines input/output transformations for SageMaker compatibility

import numpy as np
import json
from PIL import Image
import io

def input_fn(request_body, request_content_type):
    """
    Parse input data payload and convert to numpy array for model prediction.
    """
    if request_content_type == 'application/x-image':
        image = Image.open(io.BytesIO(request_body))
        image = image.resize((224, 224))  # Adjust size as needed
        arr = np.array(image) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr.astype(np.float32)
    elif request_content_type == 'application/json':
        data = json.loads(request_body)
        return np.array(data['instances']).astype(np.float32)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def output_fn(prediction, response_content_type):
    """
    Format prediction output for SageMaker response.
    """
    if response_content_type == 'application/json':
        return json.dumps({'predictions': prediction.tolist()})
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
