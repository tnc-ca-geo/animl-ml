#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    # shift 1
    # printenv
    # ls /opt
    echo "Starting TensorFlow Serving..."
    tensorflow_model_server --rest_api_port=8501 --model_name=irc --model_base_path=/opt/ml/model/irc &
    echo "Starting FastAPI server..."
    python serve.py
else
    eval "$@"
fi