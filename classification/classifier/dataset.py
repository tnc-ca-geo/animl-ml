'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
import json
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


class CTDataset(Dataset):

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.crops_dir = cfg['crops_dir']
        self.split = split
        self.transform = Compose([
            Resize((cfg['image_size'])),
            ToTensor()                          
        ])

        # don't perform augmentations on test or val splits 
        # (although at the moment they are the same)
        if split != 'train':
            self.transform = Compose([
            Resize((cfg['image_size'])),
            ToTensor()                          
        ])
        
        # index data into list
        self.data = []

        # load annotation file
        annoPath = os.path.join(
            self.data_root,
            'train_cct.json' if self.split=='train' else 'val_cct.json' # TODO: put this in config? also doesn't support making predictions against test split?
        )
        meta = json.load(open(annoPath, 'r'))

        # image id to filename lookup
        img_id_to_filename = {meta['images'][i]['id']:meta['images'][i]['file_name'] for i in range(len(meta['images']))}
        # custom labelclass indices that start at zero
        labels = {c['id']:idx for idx, c in enumerate(meta['categories'])}
        # print(f'label id-to-index: {labels}')
        
        # since we're doing classification, we're just taking the first annotation per crop and drop the rest
        # (i.e., we expect a one-to-one relationship between crops and annotations)
        images_covered = set()      # all those images for which we have already assigned a label
        for anno in meta['annotations']:
            img_id = anno['image_id']
            if img_id in images_covered:
                continue
            
            # append image_file_name-label tuple to data
            img_file_name = img_id_to_filename[img_id]
            label = anno['category_id']
            labelIndex = labels[label]
            self.data.append([img_file_name, labelIndex])
            images_covered.add(img_id)       # make sure image is only added once to dataset
    

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]              # see line 57 above where we added these two items to the self.data list

        # load image
        image_path = os.path.join(self.data_root, self.crops_dir, image_name) 
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order
        
        # transform
        img_tensor = self.transform(img)
                
        return img_tensor, label, image_path


def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = CTDataset(cfg, split)

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=(split == 'train'),
            num_workers=cfg['num_workers']
        )
    return dataLoader