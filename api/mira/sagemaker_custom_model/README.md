### Local Testing (in development, required to deploy custom container of MIRA on Sagemaker Serverless, an upgrade from Realtime endpoints use dby the above instructions)

Get the model and inference code for MIRA-Large
```
bash download_untar_model.sh
```

Build the custom docker container we need to use Sagemaker Serverless and for local testing. this installs pillow and numpy in the sagemaker container for TF 1.14.

```
docker build -t test/tf-1.14-pillow-nump:latest .
```

Start the model server with docker.

```
bash docker_mira.sh
```

Attempt to send a request following this example that base64 encodes https://www.tensorflow.org/tfx/serving/api_rest 

```
python local_test.py
```
