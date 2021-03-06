# MIRA API
API for real-time MIRA species classification

## `Intro` :fox_face:
MIRA is a pair of species classification models trained on camera trap data from 
the Channel Islands in California. The models were developed by 
[Nanolayers](http://www.nanolayers.com/). The "mira-small" model is designed to 
detect rodents, while the "mira-large" model classifies foxes and skunks.  

This API allows users to submit an image file (or the URL of an image available 
on the internet) for inference against the two MIRA models and recieve 
predictions in the response. 

### Related repos
- Mira                http://github.com/tnc-ca-geo/mira
- Mira web server     https://github.com/fullmetalfelix/Mira
- Mira worker         https://github.com/fullmetalfelix/Mira-Worker


## `Usage` :rat:
Please send inference requests in a mulitpart form. You may submit images 
either in full as binary files, or, if you'd like to run inference against an 
image hosted somewhere online, you may submit just its URL.

You can also optionally pass in an object bounding box, which the API will use 
to crop the image before sumitting it to the models for inference.

The possible parts of the form are:
- **image**: a binary image file
- **url**: a url pointing to an image online
- **bbox**: a string represntation of a bounding box in the format: 
  '[ymin, xmin, ymax, xmax]', where values are relative and the 
  origin in the upper-left

### Invocation example (Python)
See example below, or check out example useage in 
```[animl-ml/notebooks/test-inference-pipeline.ipynb](https://github.com/tnc-ca-geo/animl-ml/blob/master/notebooks/test-inference-pipeline.ipynb)``` or 
```[animl-ml/api/mira/test/test_api.py](https://github.com/tnc-ca-geo/animl-ml/blob/master/api/mira/test/test_api.py)```. 

```python
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

API_URL = 'https://9q9u0ns6ul.execute-api.us-west-2.amazonaws.com/dev/classify'
IMG_PATH = '/path/to/local/image.jpg'
# IMG_URL = 'http://www.example.com/image.jpg'
BBOX = [0.536007, 0.434649, 0.635773, 0.543599]

fields = {}
if IMG_URL:
    fields['url'] = IMG_URL
if IMG_PATH:
    fields['image'] = (IMG_PATH, open(IMG_PATH, 'rb'), 'image/jpeg')
if BBOX:
    fields['bbox'] = json.dumps(BBOX)

multipart_data = MultipartEncoder(fields=fields)
r = requests.post(API_URL,
                  data = multipart_data,
                  headers = {'Content-Type': multipart_data.content_type})
```

Prediction responses will look like:

```json
[{
  "mira-large": {
    "endpoint_name": "mira-large",
    "classes": ["fox", "skunk", "empty"],
    "predictions": {
      "fox": 4.20211e-06,
      "skunk": 0.964001,
      "empty": 0.0359947
    }
  },
  "mira-small": {
    "endpoint_name": "mira-small",
    "classes": ["rodent", "empty"],
    "predictions": {
      "rodent": 0.737944,
      "empty": 0.262056
    }
  }
}]
```

## `Development` :camera:

### Prerequisits
The commands below require you have the following tools globally installed:
- [Serverless](https://www.serverless.com/framework/docs/getting-started/)
- [aws-cli](https://aws.amazon.com/cli/)

As well as an AWS config profile called "serverless-admin". Good instructions 
[here](https://www.serverless.com/framework/docs/providers/aws/guide/credentials/).

### Deploy to dev
```shell
$ npm run deploy-dev
```

### Deploy to production
```shell
$ npm run deploy-prod
```

### Test
To test the hosted dev endpoint, make sure virtual env is activated, and either 
run the test script ```test/test_api.py``` with the ```--img [filename.jpg]``` 
option (to test submitting an image present in the ```animl-ml/input``` 
directory), or run it with the ```--img-url [http://some-image-url.jpg]``` 
option to test submitting an image that's accessible somewhere on the internet 
via url. You can also optionally pass in a bounding box.

```shell
# from the animl-ml/api/mira directory
# setup venv if you havene't already

$ virtualenv env -p python3
$ pip3 install -r requirements.txt

# activate it

$ source env/bin/activate

# to submit a local image for testing,
# image must be in animl-ml/input directory
# and use --img option:

$ python test/test_api.py \
  --img sample-img.jpg \
  --bbox "[0.536007, 0.434649, 0.635773, 0.543599]"

# OR pass a valid image from the internet via URL
# with the --img-url option:

$ python test/test_api.py \
  --img-url https://animl-sample-images.s3-us-west-1.amazonaws.com/p_001205.jpg \
  --bbox "[0.5383, 0.437, 0.63283, 0.5448999999999999]"

```


