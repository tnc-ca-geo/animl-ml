# MIRA API :rat:
Lambda handler for MIRA inference requests

## `Intro`

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
To test the hosted dev endpoint, make sure virtual env is activated, and you 
can either run the test script with the ```--img [filename.jpg]``` option to test submitting 
an image present in the ```animl-ml/input``` directory, or run it with the 
```--img-url [http://some-image-url.jpg]``` option to test submitting an image 
via url. You can also optionally pass in a bounding box.

```
# Setup venv

$ virtualenv env -p python3
$ source env/bin/activate

# to submit a local image for testing,
# image must be in animl-ml/input directory
# and use --img option:

$ python test/test_api.py --img sample-img.jpg --bbox "[0.536007, 0.434649, 0.635773, 0.543599]"

# OR pass a valid image from the internet via URL
# with the --img-url option:

$ python test/test_api.py \
  --img-url https://animl-test-images.s3-us-west-1.amazonaws.com/p_001205.jpg \
  --bbox "[0.5383, 0.437, 0.63283, 0.5448999999999999]"

```


