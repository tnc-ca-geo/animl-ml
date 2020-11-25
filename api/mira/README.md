# MIRA API :rat:
Lambda handler for MIRA inference requests

## `Intro`

## `Development`

### Setup venv
```
$ virtualenv env -p python3
$ source env/bin/activate
```

### Run locally
```
$ npm start
```

### Deploy to dev
```
$ npm run deploy-dev
```

### Deploy to production
```
$ npm run deploy-prod
```

### Test
```
# activate virtual env
$ npm start
$ python test/test_api.py sample-img.jpg
```