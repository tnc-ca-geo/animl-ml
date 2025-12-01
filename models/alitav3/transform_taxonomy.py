#!/usr/bin/env python3
import json
import csv
import random
import argparse
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

def build_taxonomy_field(row: dict) -> str:
    """Build taxonomy string from taxonomic levels in the CSV row.
    
    Args:
        row (dict): A dictionary representing a row from the CSV file.
        
    Returns:
        str: A semicolon-separated taxonomy string.
    """
    taxonomy_parts = [
        row['level_class'].split()[1].lower(),
        row['level_order'].split()[1].lower(),
        row['level_family'].split()[1].lower(),
        row['level_genus'].split()[1].lower(),
        row['level_species'].split()[2].lower()
    ]
    return ';'.join(taxonomy_parts)

def transform_taxonomy(input_file: str = 'original-model/taxon-mapping.csv', output_file: str = 'exported-model/alitav3-mongodb-record.json'):
    """Transform Alita v3 taxonomy CSV to MongoDB document format.
        
    Args:
        input_file (str): Path to input taxonomy file
        output_file (str): Path to output JSON file
    """
    
    output_data = {
        "_id": "alitav3",
        "description": "Alita v3.03 New Zealand Species Classifier",
        "version": "v3.03",
        "defaultConfThreshold": 0.5,
        "categories": []
    }
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use model_class as both _id and name
            _id = row['model_class']

            # Clean name (remove '.' and '$' if present)
            name = row['model_class'].replace('.', '').replace('$', '')

            # Build taxonomy string from taxonomic levels
            taxonomy = build_taxonomy_field(row)

            category = {
                "_id": _id,
                "name": name,
                "color": random.choice(COLORS),
                "taxonomy": taxonomy
            }
            
            output_data["categories"].append(category)
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Created MongoDB record with {len(output_data['categories'])} categories")

if __name__ == "__main__":
    transform_taxonomy()

def main():
    parser = argparse.ArgumentParser(description='Transform taxonomy data to JSON format')
    parser.add_argument('--input_file', required=False, help='Path to input taxonomy file. This is the "original-model/taxon-mapping.csv" file')
    parser.add_argument('--output_file', required=False, help='Path to output JSON file')

    args = parser.parse_args()

    if not args.input_file:
        args.input_file = 'original-model/taxon-mapping.csv'
    if not args.output_file:
        args.output_file = 'exported-model/alitav3-mongodb-record.json'

    # Ensure taxon mapping file exists
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
