#!/usr/bin/env python3
"""
Test script for Alita v3 TorchServe endpoint
"""

import base64
import json
import requests
import argparse
from pathlib import Path

def test_inference(image_path, bbox=None, endpoint_url="http://127.0.0.1:8080/invocations"):
    """
    Test the Alita v3 model endpoint with an image
    
    Args:
        image_path: Path to test image
        bbox: Bounding box as [ymin, xmin, ymax, xmax] in relative coordinates
        endpoint_url: TorchServe endpoint URL
    """
    
    # Default to full image if no bbox provided
    if bbox is None:
        bbox = [0, 0, 1, 1]
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    img_string = base64.b64encode(image_data).decode('utf-8')
    
    # Create payload
    payload = {
        "image": img_string,
        "bbox": bbox
    }
    
    # Send request
    print(f"Testing with image: {image_path}")
    print(f"Bounding box: {bbox}")
    print(f"Endpoint: {endpoint_url}")
    
    try:
        response = requests.post(
            endpoint_url,
            files={'body': json.dumps(payload)},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== Prediction Results ===")
            
            # Sort by confidence and show top 5
            if isinstance(result, list) and len(result) > 0:
                predictions = result[0]
            else:
                predictions = result
                
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            print("Top 5 predictions:")
            for i, (species, confidence) in enumerate(sorted_preds[:5]):
                print(f"{i+1:2d}. {species:<20} {confidence:.4f}")
                
        else:
            print(f"Error: HTTP {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test Alita v3 TorchServe endpoint')
    parser.add_argument('image_path', help='Path to test image')
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('ymin', 'xmin', 'ymax', 'xmax'),
                       help='Bounding box coordinates (relative, 0-1)')
    parser.add_argument('--endpoint', default='http://127.0.0.1:8080/invocations',
                       help='TorchServe endpoint URL')
    
    args = parser.parse_args()
    
    # Validate image path
    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    test_inference(args.image_path, args.bbox, args.endpoint)

if __name__ == '__main__':
    main()