# MIRA API :rat:
API for MIRA species classification requests

## `Intro`

## `Usage`

### Invocation example (Python)

### Invocation example (Node)

## `Development`

### Deploy to dev
```
$ npm run deploy-dev
```

### Deploy to production
```
$ npm run deploy-prod
```

### Test
To test the hosted dev endpoint, make sure virtual env is activated, and either 
run the test script ```test/test_api.py``` with the ```--img [filename.jpg]``` 
option (to test submitting an image present in the ```animl-ml/input``` 
directory), or run it with the ```--img-url [http://some-image-url.jpg]``` 
option to test submitting an image that's accessible somewhere on the internet 
via url. You can also optionally pass in a bounding box.

```
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
  --img-url https://animl-test-images.s3-us-west-1.amazonaws.com/p_001205.jpg \
  --bbox "[0.5383, 0.437, 0.63283, 0.5448999999999999]"

```


