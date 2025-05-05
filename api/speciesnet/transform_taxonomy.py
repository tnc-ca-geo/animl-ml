#!/usr/bin/env python3
import json
import argparse
import random
from pathlib import Path

COLORS = [
    '#8D8D8D', '#E54D2E', '#E5484D', '#E54666',
    '#E93D82', '#D6409F', '#AB4ABA', '#8E4EC6',
    '#6E56CF', '#5B5BD6', '#3E63DD', '#0090FF',
    '#00A2C7', '#12A594', '#29A383', '#30A46C',
    '#46A758', '#A18072', '#978365', '#AD7F58',
    '#F76B15', '#FFC53D', '#FFE629', '#BDEE63',
    '#86EAD4', '#7CE2FE'
]

def transform_taxonomy(input_file: str, output_file: str):
    """
    Transform taxonomy data from semicolon-separated format to JSON objects.
    
    Args:
        input_file (str): Path to input taxonomy file
        output_file (str): Path to output JSON file
    """
    output_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue

            # Split the line by semicolons
            parts = line.strip().split(';')
            
            # Create JSON object
            taxonomy_obj = {
                "_id": parts[0],
                "taxonomy": ";".join(parts[1:-1]),
                "name": parts[-1],
                "color": random.choice(COLORS)
            }
            
            output_data.append(taxonomy_obj)
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Transform taxonomy data to JSON format')
    parser.add_argument('input_file', help='Path to input taxonomy file. This is the "always_crop...labels.txt" file')
    parser.add_argument('output_file', help='Path to output JSON file')
    
    args = parser.parse_args()
    
    # Ensure input file exists
    if not Path(args.input_file).is_file():
        print(f"Error: Input file '{args.input_file}' does not exist")
        return
        
    try:
        transform_taxonomy(args.input_file, args.output_file)
        print(f"Successfully transformed data to {args.output_file}")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
