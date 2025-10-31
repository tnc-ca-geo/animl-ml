#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    echo "Starting TensorFlow Serving..."
    tensorflow_model_server --rest_api_port=8501 --model_name=irc --model_base_path=/opt/ml/model/irc &
    
    # Wait for TensorFlow Serving to be ready
    echo "Waiting for TensorFlow Serving to be ready..."
    until curl -s http://localhost:8501/v1/models/irc > /dev/null; do
        sleep 1
    done
    echo "TensorFlow Serving is ready."

    echo "Starting FastAPI server..."
    python serve.py
else
    exec "$@"
fi