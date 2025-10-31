"""
TODO: UPDATE 

Adapted from crop_detections.py in microsoft/CameraTraps/classification
https://github.com/microsoft/CameraTraps

Given a COCO for Cameratraps JSON file from as input, crops the bounding boxes.

We assume that no image contains over 100 bounding boxes, and we always save
crops as RGB .jpg files for consistency.

Example cropped image path:
    "path/to/image.jpg___crop00.jpg"

By default, the images are cropped exactly per the given bounding box
coordinates. However, if square crops are desired, pass the --square-crops
flag. This will always generate a square crop whose size is the larger of the
bounding box width or height. In the case that the square crop boundaries exceed
the original image size, the crop is padded with 0s.

This script outputs a log file to
    <output_dir>/crop_detections_log_{timestamp}.json
which contains images that failed to download and crop properly.

Example command:

python crop_detections.py \
    detections.json \
    /path/to/crops \
    --images-dir /path/to/images \
    --square-crops \
    --threads 50 \
    --logdir "."
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from concurrent import futures
from datetime import datetime
import io
import json
import os
from typing import Any, BinaryIO, Optional
from collections import defaultdict
import traceback

from PIL import Image, ImageOps, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(cct_json_path: str,
         cropped_images_dir: str,
         images_dir: Optional[str],
         save_full_images: bool,
         crop_strategy: str,
         check_crops_valid: bool,
         threads: int,
         logdir: str) -> None:
    """
    Args:
        cct_json_path: str, path to COCO for Camertraps JSON file
        cropped_images_dir: str, path to local directory for saving crops of
            bounding boxes
        images_dir: optional str, path to local directory where images are saved
        save_full_images: bool, whether to save downloaded images to images_dir,
            images_dir must be given if save_full_images=True
        crop_strategy: str, strategy for making corps square
        check_crops_valid: bool, whether to load each crop to ensure the file is
            valid (i.e., not truncated)
        threads: int, number of threads to use for downloading images
        logdir: str, path to directory to save log file
    """
    # error checking
    if save_full_images:
        assert images_dir is not None, \
            'save_full_images specified but no images_dir provided'
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)
            print(f'Created images_dir at {images_dir}')

    print(f'crop strategy: {crop_strategy}')

    # load CCT JSON
    with open(cct_json_path, 'r') as f:
        cct = json.load(f)

    images_failed, num_crops = download_and_crop(
        cct=cct,
        cropped_images_dir=cropped_images_dir,
        images_dir=images_dir,
        save_full_images=save_full_images,
        crop_strategy=crop_strategy,
        check_crops_valid=check_crops_valid,
        threads=threads)
    print(f'{len(images_failed)} images failed to crop.')

    # save log of bad images
    log = {
        'images_failed_to_crop': images_failed,
        'num_new_crops': num_crops
    }
    os.makedirs(logdir, exist_ok=True)
    date = datetime.now().strftime('%Y%m%d_%H%M%S')  # e.g., '20200722_110816'
    log_path = os.path.join(logdir, f'crop_detections_log_{date}.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=1)


def download_and_crop(
        cct: Mapping[str, Any],
        cropped_images_dir: str,
        images_dir: Optional[str],
        save_full_images: bool,
        crop_strategy: str,
        check_crops_valid: bool,
        threads: int = 1
        ) -> tuple[list[str], int, int]:
    """
    Saves crops to a file with the same name as the original image with an
    additional suffix appended, starting with 3 underscores: "___cropXX.jpg",
    where "XX" indicates the bounding box index

    See module docstring for more info and examples.

    Args:
        detections: dict, maps image paths to info dict
            {
                "detections": [{
                    "category": "animal",  # must be name, not "1" or "2"
                    "conf": 0.926,
                    "bbox": [0.0, 0.2762, 0.1539, 0.2825],
                }],
                "is_ground_truth": True  # whether bboxes are ground truth
            }
        cct: coco for cameratraps formatted json object
        cropped_images_dir: str, path to folder where cropped images are saved
        images_dir: optional str, path to folder where full images are saved
        save_full_images: bool, whether to save downloaded images to images_dir,
            images_dir must be given and must exist if save_full_images=True
        crop_strategy: str, strategy for making crops square
        check_crops_valid: bool, whether to load each crop to ensure the file is
            valid (i.e., not truncated)
        threads: int, number of threads to use for downloading images

    Returns:
        images_failed: list of str, images with bounding boxes that
            failed to crop properly
        total_new_crops: int, number of new crops saved to cropped_images_dir
    """
    # always save as .jpg for consistency
    crop_path_template = os.path.join(
        cropped_images_dir, '{img_path}___crop_{anno_id}.jpg')

    pool = futures.ThreadPoolExecutor(max_workers=threads)
    future_to_img_path = {}
    images_failed = []
    
    print(f'Builing map of image ids to annotations...')
    image_id_to_annotations = defaultdict(list)
    for ann in tqdm(cct['annotations']):
        image_id_to_annotations[ann['image_id']].append(ann)

    print(f'Getting bbox info for {len(cct["annotations"])} annotations in {len(cct["images"])} iamges...')

    # iterate over images
    for img in tqdm(cct['images']):
        img_path = img['file_name']
        # find all annotations for this image
        bbox_dicts = image_id_to_annotations[img['id']]

        # get the image from disk
        future = pool.submit(
            load_and_crop, img_path, images_dir, bbox_dicts,
            crop_path_template, save_full_images, crop_strategy,
            check_crops_valid)
        future_to_img_path[future] = img_path

    total = len(future_to_img_path)
    total_new_crops = 0
    print(f'Reading {total} images and cropping...')
    for future in tqdm(futures.as_completed(future_to_img_path), total=total):
        img_path = future_to_img_path[future]
        try:
            num_new_crops = future.result()
            total_new_crops += num_new_crops
        except Exception as e:  # pylint: disable=broad-except
            exception_type = type(e).__name__
            tqdm.write(f'{img_path} - generated {exception_type}: {e}')
            tqdm.write("".join(traceback.TracebackException.from_exception(e).format()) == traceback.format_exc() == "".join(traceback.format_exception(type(e), e, e.__traceback__)))
            tqdm.write("".join(traceback.TracebackException.from_exception(e).format()))
            images_failed.append(img_path)

    pool.shutdown()

    print(f'Made {total_new_crops} new crops.')
    return images_failed, total_new_crops


def load_local_image(img_path: str |  BinaryIO) -> Optional[Image.Image]:
    """Attempts to load an image from a local path."""
    try:
        with Image.open(img_path) as img:
            img.load()
        return img
    except OSError as e:  # PIL.UnidentifiedImageError is a subclass of OSError
        exception_type = type(e).__name__
        tqdm.write(f'Unable to load {img_path}. {exception_type}: {e}.')
    return None


def load_and_crop(img_path: str,
                  images_dir: Optional[str],
                  bbox_dicts: Iterable[Mapping[str, Any]],
                  crop_path_template: str,
                  save_full_image: bool,
                  crop_strategy: str,
                  check_crops_valid: bool) -> tuple[bool, int]:
    """Given an image and a list of bounding boxes, checks if the crops already
    exist. If not, loads the image locally, then crops it.

    local image path: <images_dir>/<img_path>

    Args:
        img_path: str, image path
        images_dir: optional str, path to local directory of images, and where
            full images are saved if save_full_images=True
        bbox_dicts: list of dicts, each dict contains info on a bounding box
        crop_path_template: str, contains placeholders {img_path} and {anno_id}
        save_full_images: bool, whether to save downloaded images to images_dir,
            images_dir must be given and must exist if save_full_images=True
        check_crops_valid: bool, whether to load each crop to ensure the file is
            valid (i.e., not truncated)

    Returns:
        num_new_crops: int, number of new crops successfully saved
    """
    num_new_crops = 0

    # crop_path => normalized bbox coordinates [xmin, ymin, width, height]
    bboxes_tocrop: dict[str, list[float]] = {}
    for i, bbox_dict in enumerate(bbox_dicts):
        crop_path = crop_path_template.format(img_path=img_path, anno_id=bbox_dict['id'])
        if not os.path.exists(crop_path) or (
                check_crops_valid and load_local_image(crop_path) is None):
            bboxes_tocrop[crop_path] = bbox_dict['bbox']
    if len(bboxes_tocrop) == 0:
        return num_new_crops

    img = None

    # try loading image from local directory
    if images_dir is not None:
        full_img_path = os.path.join(images_dir, img_path)
        debug_path = full_img_path
        if os.path.exists(full_img_path):
            img = load_local_image(full_img_path)

    assert img is not None, 'image "{}" failed to load properly'.format(
        debug_path)
    
    if img.mode != 'RGB':
        img = img.convert(mode='RGB')  # always save as RGB for consistency

    # crop the image
    for crop_path, bbox in bboxes_tocrop.items():
        num_new_crops += save_crop(
            img, bbox=bbox, crop_strategy=crop_strategy, save=crop_path)
    return num_new_crops


def save_crop(img: Image.Image, bbox: Sequence[float], crop_strategy: str,
              save: str) -> bool:
    """Crops an image and saves the crop to file.

    Args:
        img: PIL.Image.Image object, already loaded
        bbox: list or tuple of float, [x,y,width,height] (absolute, origin upper-left)
        crop_strategy: str, stragegy for making crops square
        save: str, path to save cropped image

    Returns: bool, True if a crop was saved, False otherwise
    """

    img_w, img_h = img.size
    xmin = bbox[0]
    ymin = bbox[1]
    box_w = bbox[2]
    box_h = bbox[3]

    # if crop_strategy is 'square' or 'pad', determine size of square
    box_size = max(box_w, box_h)
    if type(box_size) is float:
        box_size = int(box_size)

    if crop_strategy == 'square':
        xmin = max(0, min(
            xmin - int((box_size - box_w) / 2),
            img_w - box_w))
        ymin = max(0, min(
            ymin - int((box_size - box_h) / 2),
            img_h - box_h))
        box_w = min(img_w, box_size)
        box_h = min(img_h, box_size)

    if box_w == 0 or box_h == 0:
        tqdm.write(f'Skipping size-0 crop (w={box_w}, h={box_h}) at {save}')
        return False

    # Image.crop() takes box=[left, upper, right, lower]
    crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])

    if ((crop_strategy == 'square' and (box_w != box_h)) or
        crop_strategy == 'pad'):
        crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)

    os.makedirs(os.path.dirname(save), exist_ok=True)
    crop.save(save)
    return True


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='COCO for Cameratraps file.')
    parser.add_argument(
        'cct_json',
        help='path to COCO for Cameratraps JSON file')
    parser.add_argument(
        'cropped_images_dir',
        help='path to local directory for saving crops of bounding boxes')
    parser.add_argument(
        '-i', '--images-dir',
        help='path to directory where full images are already available, '
             'or where images will be written if --save-full-images is set')
    parser.add_argument(
        '--save-full-images', action='store_true',
        help='forces downloading of full images to --images-dir')
    parser.add_argument(
        '--crop-strategy', choices=['square', 'pad'], default='square',
        help='strategy for making crops square')
    parser.add_argument(
        '--check-crops-valid', action='store_true',
        help='load each crop to ensure file is valid (i.e., not truncated)')
    parser.add_argument(
        '-n', '--threads', type=int, default=1,
        help='number of threads to use for downloading and cropping images')
    parser.add_argument(
        '--logdir', default='.',
        help='path to directory to save log file')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(cct_json_path=args.cct_json,
         cropped_images_dir=args.cropped_images_dir,
         images_dir=args.images_dir,
         save_full_images=args.save_full_images, # TODO: remove save_full_images arg. we don't need this.
         crop_strategy=args.crop_strategy,
         check_crops_valid=args.check_crops_valid,
         threads=args.threads,
         logdir=args.logdir)