# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import logging
import json
from collections import namedtuple
import os
import boto3
import numpy as np
from PIL import Image

BUCKET = 'animl-images'
log = logging.getLogger(__name__)

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_uri, '
                     'custom_attributes, request_content_type, accept_header')

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details

    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """

    log.info('logging - input_handler firing')

    if context.request_content_type == 'application/json':
        d = data.read().decode('utf-8')
        log.info('request recieved: {}'.format(d))
        # Download image from s3
        s3 = boto3.client('s3')
        result = s3.get_bucket_acl(Bucket=BUCKET)
        log.info('bucket acl: {}'.format(result))
        with open('s3_img.jpg', 'wb') as f:
            s3.download_fileobj(BUCKET, d, f)
            img=Image.open('s3_img.jpg')
            np_img=np.asarray(img, np.uint8)
            np_img_expanded = np.expand_dims(np_img, axis=0)
            log.info('image shape: {}'.format(np_img_expanded.shape))
            serialized = json.dumps({'signature_name': 'serving_default', 'instances': np_img_expanded.tolist()})
            return serialized
        # # pass through json (assumes it's correctly formed)
        # print('content type is application/json')
        # log.info('logging - content type is application/json')
        # d = data.read().decode('utf-8')
        # return d if len(d) else ''

    if context.request_content_type == 'text/csv':
        # very simple csv handler
        return json.dumps({
            'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.

    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details

    Returns:
        (bytes, string): data to return to client, response content type
    """
    print('print - output_handler firing')
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type
