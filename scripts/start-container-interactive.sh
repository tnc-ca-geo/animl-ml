#!/bin/bash
#
# Start a local docker container in interactive mode

set -euo pipefail

MODEL_DIR="$(cd "models" > /dev/null && pwd)"
full_version=1.13.0
device=cpu

docker run \
    -v "$MODEL_DIR":/opt/ml/model:ro \
    -p 8080:8080 \
    -e "SAGEMAKER_TFS_DEFAULT_MODEL_NAME=saved_model_megadetector_v3_tf19" \
    -e "SAGEMAKER_TFS_NGINX_LOGLEVEL=error" \
    -e "SAGEMAKER_BIND_TO_PORT=8080" \
    -e "SAGEMAKER_SAFE_PORT_RANGE=9000-9999" \
    -e "AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID \
    -e "AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY \
    -e "AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN \
    -e "AWS_SECURITY_TOKEN="$AWS_SECURITY_TOKEN \
    -it tensorflow-inference:$full_version-$device bin/bash

