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

# Step 4: Run container
echo "Starting container..."
bash docker-run.sh $(pwd)/exported-model &

# Wait for container to start
echo "Waiting for container to start..."
sleep 10

# Step 5: Test with curl
echo "Testing model with sample request..."

# Create a simple test (you'll need to replace with actual image)
echo "To test the model, use the following curl command with a base64-encoded image:"
echo ""
echo "IMG_STRING=\$(base64 -i /path/to/test/image.jpg) \\"
echo "BBOX=[0,0,1,1] \\"
echo "PAYLOAD=\$( jq -n \\"
echo "            --arg image \"\$IMG_STRING\" \\"
echo "            --arg bbox \"\$BBOX\" \\"
echo "            '{image: \$image, bbox: \$bbox}' )"
echo ""
echo "curl -i http://127.0.0.1:8080/invocations -F body=\$PAYLOAD"

echo ""
echo "=== Build and setup complete ==="
echo "Container is running on http://127.0.0.1:8080"
echo "To stop the container, run: docker stop \$(docker ps -q --filter ancestor=alitav3:latest-cpu)"