"""Test script to verify SpeciesNet server functionality."""
# Run after building and running the Docker container: Dockerfile.sagemaker

import base64
import json
import requests
from typing import Optional, Dict, Any
from PIL import Image
from io import BytesIO
from speciesnet import draw_bboxes
from fixtures.responses import (
    HEALTH_CHECK_RESPONSE,
    DEFAULT_SETTINGS_RESPONSE,
    CLASSIFIER_ONLY_RESPONSE,
    CLASSIFIER_WITH_BBOX_RESPONSE,
    DETECTOR_ONLY_RESPONSE,
    NO_GEOFENCE_RESPONSE
)

def read_test_image(filepath: str = 'test_data/african_elephants.jpg') -> str:
    """Read and encode test image."""
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')

def make_request(
    image_base64: str,
    *,
    components: Optional[str] = None,
    geofence: Optional[bool] = None,
    batch_size: Optional[int] = None,
    country: Optional[str] = None,
    bbox: Optional[list[float]] = None
) -> Dict[str, Any]:
    """Make a request to the SpeciesNet server with given parameters."""
    # Build URL with query parameters
    url = 'http://localhost:8080/invocations'
    params = {}
    if components is not None:
        params['components'] = components
    if geofence is not None:
        params['geofence'] = str(geofence).lower()
    if batch_size is not None:
        params['batch_size'] = batch_size

    # Build payload
    payload = {"image_data": image_base64}
    if country is not None:
        payload["country"] = country
    if bbox is not None:
        payload["bbox"] = bbox

    # Make request
    response = requests.post(url, params=params, json=payload)
    response.raise_for_status()
    return response.json()

def test_health_check():
    """Test the health check endpoint."""
    response = requests.get('http://localhost:8080/ping')
    print("\nHealth check response:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200
    assert response.json() == HEALTH_CHECK_RESPONSE
    print("Health check test: PASSED")

def test_default_settings(image_base64):
    """Test API with default settings (all components enabled)."""
    print("\nTesting default settings...")
    result = make_request(image_base64, country="KEN")
    print("Default settings response:")
    print(json.dumps(result, indent=2))
    expected_classes_count = len(DEFAULT_SETTINGS_RESPONSE["predictions"][0]["classifications"]["classes"])
    expected_detections_count = len(DEFAULT_SETTINGS_RESPONSE["predictions"][0]["detections"])
    assert len(result["predictions"][0]["classifications"]["classes"]) == expected_classes_count
    assert len(result["predictions"][0]["detections"]) == expected_detections_count

def test_classifier_only(image_base64):
    """Test API with classifier component only."""
    print("\nTesting classifier only...")
    result = make_request(
        image_base64,
        components="classifier",
        batch_size=16,
        country="KEN"
    )
    print("Classifier only response:")
    print(json.dumps(result, indent=2))
    expected_classes_count = len(CLASSIFIER_ONLY_RESPONSE["predictions"][0]["classifications"]["classes"])
    assert len(result["predictions"][0]["classifications"]["classes"]) == expected_classes_count
    assert "detections" not in result["predictions"][0]

def test_detector_only(image_base64):
    """Test API with detector component only."""
    print("\nTesting detector only...")
    result = make_request(
        image_base64,
        components="detector",
        country="KEN"
    )
    print("Detector only response:")
    print(json.dumps(result, indent=2))
    assert "classifications" not in result["predictions"][0]
    expected_detections_count = len(DETECTOR_ONLY_RESPONSE["predictions"][0]["detections"])
    assert len(result["predictions"][0]["detections"]) == expected_detections_count

def test_classifier_with_bbox(image_base64):
    """Test API with classifier component and specified bbox."""
    print("\nTesting classifier with bbox...")

    # Decode and load image
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes))

    # Use first detected bbox from DETECTOR_ONLY_RESPONSE
    first_detection = DETECTOR_ONLY_RESPONSE["predictions"][0]["detections"][1]
    bbox = first_detection["bbox"]

    print("Using bbox:", bbox)

    # Draw bboxes on image copy to test
    image_copy = image.copy()
    image_copy = draw_bboxes(image_copy, DETECTOR_ONLY_RESPONSE["predictions"][0]["detections"])
    image_copy.save("test_output_with_bbox.png")

    result = make_request(
        image_base64,
        components="classifier",
        bbox=bbox
    )
    print("Classifier with bbox response:")
    print(json.dumps(result, indent=2))

    expected_classes_count = len(CLASSIFIER_WITH_BBOX_RESPONSE["predictions"][0]["classifications"]["classes"])
    assert len(result["predictions"][0]["classifications"]["classes"]) == expected_classes_count
    assert "detections" not in result["predictions"][0]

def test_no_geofencing(image_base64):
    """Test API with geofencing disabled."""
    print("\nTesting without geofencing...")
    result = make_request(
        image_base64,
        geofence=False
    )
    print("No geofencing response:")
    print(json.dumps(result, indent=2))
    expected_classes_count = len(NO_GEOFENCE_RESPONSE["predictions"][0]["classifications"]["classes"])
    expected_detections_count = len(NO_GEOFENCE_RESPONSE["predictions"][0]["detections"])
    assert len(result["predictions"][0]["classifications"]["classes"]) == expected_classes_count
    assert len(result["predictions"][0]["detections"]) == expected_detections_count
