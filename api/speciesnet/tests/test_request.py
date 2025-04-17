# Test script to verify SpeciesNet server functionality.
# Run after building and running the Docker container: Dockerfile.sagemaker

import base64
import json
import requests

# Read the test image
with open('test_data/african_elephants.jpg', 'rb') as f:
    image_bytes = f.read()

# Convert to base64
image_base64 = base64.b64encode(image_bytes).decode('utf-8')

# Create payload
payload = {
    "image_data": image_base64,
    "country": "KEN"  # Optional parameter
}

# Send request
response = requests.post('http://localhost:8080/invocations', json=payload)
print(json.dumps(response.json(), indent=2))
