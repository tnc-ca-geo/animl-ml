#!/bin/bash

# Build and test script for Alita v3 model

set -e

echo "=== Building Alita v3 TorchServe Model ==="

# Step 1: Install torch-model-archiver if not already installed
echo "Installing torch-model-archiver..."
pip install torch-model-archiver

# Step 2: Create the model archive
echo "Creating model archive..."
torch-model-archiver \
    --model-name alitav3 \
    --version 3.0.3 \
    --serialized-file exported-model/alitav3_compiled_cpu.pt \
    --extra-files exported-model/index_to_name.json \
    --handler alitav3_handler.py

# Move the .mar file to exported-model directory
mv alitav3.mar exported-model/alitav3.mar

echo "Model archive created: exported-model/alitav3.mar"

# Step 3: Build Docker image
echo "Building Docker image..."
docker build -t alitav3:latest-cpu .

echo "Docker image built: alitav3:latest-cpu"