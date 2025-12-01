#!/usr/bin/env python3
"""
Compare original Alita model predictions with containerized version predictions
"""

import pandas as pd
import json
import requests
import base64
import argparse
from pathlib import Path
import ast

def convert_bbox_format(x_min, y_min, width, height):
    """Convert bbox from CSV format to container format"""
    # CSV: [x_min, y_min, width, height] -> Container: [ymin, xmin, ymax, xmax]
    ymin = y_min
    xmin = x_min
    ymax = y_min + height
    xmax = x_min + width
    return [ymin, xmin, ymax, xmax]

def get_container_prediction(image_path, bbox, endpoint_url="http://127.0.0.1:8080/invocations"):
    """Get prediction from containerized model"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        img_string = base64.b64encode(image_data).decode('utf-8')
        payload = {"image": img_string, "bbox": bbox}
        
        response = requests.post(
            endpoint_url,
            files={'body': json.dumps(payload)},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0]
            return result
        else:
            print(f"Error: HTTP {response.status_code} for {image_path}")
            return None
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def parse_csv_predictions(row):
    """Parse top 3 predictions from CSV row"""
    predictions = {}
    
    # Get top 3 predictions and probabilities
    if pd.notna(row['Prediction']) and pd.notna(row['Probability']):
        predictions[row['Prediction']] = row['Probability']
    
    if pd.notna(row['Second_Pred']) and pd.notna(row['Second_Prob']):
        predictions[row['Second_Pred']] = row['Second_Prob']
    
    if pd.notna(row['Third_Pred']) and pd.notna(row['Third_Prob']):
        predictions[row['Third_Pred']] = row['Third_Prob']
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Compare original vs containerized Alita predictions')
    parser.add_argument('csv_path', help='Path to Exp_60_run_01_full_predictions.csv')
    parser.add_argument('images_path', help='Path to test images folder')
    parser.add_argument('--endpoint', default='http://127.0.0.1:8080/invocations',
                       help='Container endpoint URL')
    parser.add_argument('--output', default='prediction_comparison.csv',
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Load CSV
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} predictions from CSV")
    
    results = []
    
    for idx, row in df.iterrows():
        print(f"Processing {idx+1}/{len(df)}: {row['File_Path']}")
        
        # Parse original predictions
        original_preds = parse_csv_predictions(row)
        
        # Parse bbox coordinates
        try:
            x_min = row['x_min']
            y_min = row['y_min']
            width = row['Width']
            height = row['Height']
        except:
            print(f"Error parsing bbox for row {idx}: {row['File_Path']}")
            continue
        
        # Convert bbox format
        container_bbox = convert_bbox_format(x_min, y_min, width, height)
        
        # Construct full image path
        image_filename = Path(row['File_Path'])
        path_parts = image_filename.parts
        relative_image_path = '/'.join(path_parts[-2:])
        full_image_path = Path(args.images_path) / relative_image_path
        
        if not full_image_path.exists():
            print(f"Image not found: {full_image_path}")
            continue
        
        # Get container prediction
        container_preds = get_container_prediction(str(full_image_path), container_bbox, args.endpoint)
        
        if container_preds is None:
            continue
        
        # Get top 3 from container predictions
        sorted_container = sorted(container_preds.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Create result row
        result = {
            'file_path': row['File_Path'],
            'image_filename': image_filename,
            'bbox_original': f"[{x_min}, {y_min}, {width}, {height}]",
            'bbox_container': str(container_bbox),
            
            # Original predictions
            'orig_pred_1': list(original_preds.keys())[0] if len(original_preds) >= 1 else '',
            'orig_prob_1': list(original_preds.values())[0] if len(original_preds) >= 1 else 0,
            'orig_pred_2': list(original_preds.keys())[1] if len(original_preds) >= 2 else '',
            'orig_prob_2': list(original_preds.values())[1] if len(original_preds) >= 2 else 0,
            'orig_pred_3': list(original_preds.keys())[2] if len(original_preds) >= 3 else '',
            'orig_prob_3': list(original_preds.values())[2] if len(original_preds) >= 3 else 0,
            
            # Container predictions
            'container_pred_1': sorted_container[0][0] if len(sorted_container) >= 1 else '',
            'container_prob_1': sorted_container[0][1] if len(sorted_container) >= 1 else 0,
            'container_pred_2': sorted_container[1][0] if len(sorted_container) >= 2 else '',
            'container_prob_2': sorted_container[1][1] if len(sorted_container) >= 2 else 0,
            'container_pred_3': sorted_container[2][0] if len(sorted_container) >= 3 else '',
            'container_prob_3': sorted_container[2][1] if len(sorted_container) >= 3 else 0,
            
            # Comparison metrics
            'top1_match': (list(original_preds.keys())[0].lower() if len(original_preds) >= 1 else '') == (sorted_container[0][0].lower() if len(sorted_container) >= 1 else ''),
            'top1_prob_diff': abs((list(original_preds.values())[0] if len(original_preds) >= 1 else 0) - (sorted_container[0][1] if len(sorted_container) >= 1 else 0))
        }
        
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # Print summary
    if len(results) > 0:
        top1_accuracy = results_df['top1_match'].mean()
        avg_prob_diff = results_df['top1_prob_diff'].mean()
        print(f"Top-1 accuracy: {top1_accuracy:.3f}")
        print(f"Average probability difference: {avg_prob_diff:.4f}")

if __name__ == '__main__':
    main()