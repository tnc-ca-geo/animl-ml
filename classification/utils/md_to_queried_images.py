#
# md_to_queried_images.py
#
# "Converts" a MD results file to output from json_validator.py.
# Sample entry:
#
# "caltech/cct_images/59f79901-23d2-11e8-a6a3-ec086b02610b. jpg": {
#     "dataset": "caltech",
#     "location": 13,
#     "class": "mountain_lion",  // class from dataset in MegaDB
#     "bbox": [{"category": "animal",
#               "bbox": [0, 0.347, 0.237, 0.257]}],
#     "label": ["cat"]  // labels to use in classifier
# },
#

#%% Constants and imports

import argparse
import os
import json
from collections import defaultdict
from tqdm import tqdm


#%% Functions

def md_to_queried_images(input_filename,dataset,output_filename=None):

    ## Validate input

    assert os.path.isfile(input_filename)

    if (output_filename is None):

        tokens = os.path.splitext(input_filename)
        assert len(tokens) == 2
        output_filename = tokens[0] + '_queried_images' + tokens[1]


    ## Read input

    with open(input_filename,'r') as f:
        d = json.load(f)
  
    for s in ['info','detection_categories','images']:
        assert s in d.keys(), 'Cannot find category {} in input file, is this a MD detections file?'.format(s)
    
    ## Process images
    
    images_out = {}
    
    # im = d['images'][0]
    for im in tqdm(d['images']):

        # for now, only use images with a single detection
        # TODO: perhaps do this filtering earlier
        if len(im['detections']) != 1:
            continue

        catId = im['detections'][0]['category']
        label = d['detection_categories'][catId]
        bbox = {"category": "animal", "bbox": im['detections'][0]['bbox']}
    
        im_out = {}
        im_out['dataset'] = dataset
        im_out['location'] = im['location']
        im_out['class'] = label
        im_out['bbox'] = bbox,
        im_out['label'] = [label]
        
        key = f"{dataset}/{im['file']}"
        images_out[key] = im_out
        
    # ...for each image
    
    
    ## Write output
        
    with open(output_filename,'w') as f:
        json.dump(images_out, f, indent=1)
        
    return output_filename


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Converts MegaDetector output to queried_images.json file')
    parser.add_argument(
        '-i', '--input_filename', required=True,
        help='path to MegaDetector output JSON file')
    parser.add_argument(
        '-d', '--dataset', required=True,
        help='name of dataset')
    parser.add_argument(
        '-o', '--output_filename',
        help='filename for output')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    md_to_queried_images(
        input_filename=args.input_filename,
        dataset=args.dataset,
        output_filename=args.output_filename)