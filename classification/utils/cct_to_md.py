#
# Adapted from cct_to_md.py in microsoft/CameraTraps/data_management
# https://github.com/microsoft/CameraTraps
#
# "Converts" a COCO Camera Traps file to a MD results file.  Currently ignores
# non-bounding-box annotations, and gives all annotations a confidence of 1.0.
#
# Currently assumes that width and height are present in the input data, does not
# read them from images.
#

#%% Constants and imports

import argparse
import os
import json

from collections import defaultdict
from tqdm import tqdm


#%% Functions

def cct_to_md(input_filename,output_filename=None):
    ## Validate input
    
    assert os.path.isfile(input_filename)
    
    if (output_filename is None):
        
        tokens = os.path.splitext(input_filename)
        assert len(tokens) == 2
        output_filename = tokens[0] + '_md-format' + tokens[1]
    
        
    ## Read input
    
    with open(input_filename,'r') as f:
        d = json.load(f)
        
    for s in ['annotations','images','categories']:
        assert s in d.keys(), 'Cannot find category {} in input file, is this a CCT file?'.format(s)
        
    
    ## Prepare metadata
    
    image_id_to_annotations = defaultdict(list)
    
    # ann = d['annotations'][0]
    for ann in tqdm(d['annotations']):
        image_id_to_annotations[ann['image_id']].append(ann)
    
    category_id_to_name = {}
    for cat in d['categories']:
        category_id_to_name[str(cat['id'])] = cat['name']
        
    results = {}
    
    info = {}
    info['format_version'] = 1.2
    info['detector'] = 'cct_to_md'
    results['info'] = info
    results['detection_categories'] = category_id_to_name
        
    
    ## Process images
    
    images_out = []
    
    # im = d['images'][0]
    for im in tqdm(d['images']):
        
        im_out = {}
        im_out['file'] = im['original_relative_path']
        im_out['location'] = im['location']
        im_out['id'] = im['id']
        
        image_h = im['height']
        image_w = im['width']
        
        detections = []
        
        annotations_this_image = image_id_to_annotations[im['id']]
        
        max_detection_conf = 0
        
        for ann in annotations_this_image:
            
               if 'bbox' in ann:
                   
                   det = {}
                   det['category'] = str(ann['category_id'])
                   det['conf'] = 1.0
                   max_detection_conf = 1.0
                   
                   # MegaDetector: [x,y,width,height] (normalized, origin upper-left)
                   # CCT: [x,y,width,height] (absolute, origin upper-left)
                   bbox_in = ann['bbox']
                   bbox_out = [bbox_in[0]/image_w,bbox_in[1]/image_h,
                               bbox_in[2]/image_w,bbox_in[3]/image_h]
                   det['bbox'] = bbox_out
                   detections.append(det)
                   
              # ...if there's a bounding box
              
        # ...for each annotation
        
        im_out['detections'] = detections
        im_out['max_detection_conf'] = max_detection_conf
        im_out['is_ground_truth'] = True
    
        images_out.append(im_out)
        
    # ...for each image
    
    
    ## Write output
    
    results['images'] = images_out
    
    with open(output_filename,'w') as f:
        json.dump(results, f, indent=1)
        
    return output_filename

# ...cct_to_md()    


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Converts image annotations in COCO for Cameratraps format MegaDetector output format')
    parser.add_argument(
        '-i', '--input_filename', required=True,
        help='path to cct JSON file')
    parser.add_argument(
        '-o', '--output_filename',
        help='filename for output')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    cct_to_md(
        input_filename=args.input_filename,
        dataset=args.dataset,
        output_filename=args.output_filename)