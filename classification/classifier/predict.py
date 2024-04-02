'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
'''

import os
import argparse
import yaml
import glob
from tqdm import trange
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np

from util import init_seed
from dataset import create_dataloader
from model import CustomResNet18

def load_model(cfg, checkpoint, dataloader):
    '''
        Creates a model instance and loads the checkpoint state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])
    checkpoint_path = os.path.join(cfg['experiment_dir'], 'checkpoints', checkpoint)

    print(f'Loading model from checkpoint {checkpoint_path}')
    state = torch.load(open(checkpoint_path, 'rb'), map_location='cpu')
    model_instance.load_state_dict(state['model'])

    return model_instance

def test(cfg, dataLoader, model):
    '''
        Inference function. Note that this looks almost the same as the validation
        function, in train, except the metrics we save are different
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    model.eval()
    
    # set up empty lists for metrics to track/save
    true_labels = []
    pred_labels = []
    logits = []
    scores = []
    filepaths = []

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels, image_paths) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            prediction = model(data)

            # add true labels to the true labels list
            labels_np = labels.cpu().detach().numpy()
            true_labels.extend(labels_np)

            # logits are the raw output from the model (and can be and real number)
            # they're worth storing and plotting on histograms
            # softmax coerces the logits to a score between 0-1 
            logits.extend(prediction.cpu().detach().numpy().tolist())
            scores.extend(prediction.softmax(dim=1).cpu().detach().numpy().tolist())

            # add predicted labels to the predicted labels list
            pred_label = torch.argmax(prediction, dim=1)
            pred_label_np = pred_label.cpu().detach().numpy()
            pred_labels.extend(pred_label_np)

            # add filepath
            filepaths.extend(image_paths)

            progressBar.update(1)
    
    progressBar.close()
    print(f'INFERENCE DONE - generated predictions for {len(filepaths)} images')
    return pred_labels, true_labels, logits, scores, filepaths


def main(config_path, checkpoint, split):

    print(f'Using config "{config_path}"')
    experiment_dir = Path(config_path).parent.absolute()
    cfg = yaml.safe_load(open(config_path, 'r'))
    cfg['experiment_dir'] = experiment_dir
    
    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # initialize data loaders for training and validation set
    dl = create_dataloader(cfg, split=split)

    # initialize model
    model = load_model(cfg, checkpoint, dl)

    pred_labels, true_labels, logits, scores, filepaths = test(cfg, dl, model)

    output_dict = {'pred_labels': [int(i) for i in pred_labels], 
                   'true_labels': [int(i) for i in true_labels],
                   'logits': logits,
                   'scores': scores,
                   'filepaths': filepaths}

    output_dict.keys()
    output_dict['pred_labels'][0]
    out_file = os.path.join(experiment_dir, 'predictions', f'{split}_results.json')
    print(f'INFERENCE DONE - writing results to {out_file}')

    with open(out_file,'w') as f:
        json.dump(output_dict, f)      

def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Make predictions on a deep learning model.')

    parser.add_argument(
        '--config',
        default='configs/exp_resnet18.yaml',
        help='Path to config file')

    parser.add_argument(
        '--checkpoint',
        help='filename of checkpoint to evaluate (e.g. "200.pt")')

    parser.add_argument(
        '--split', choices=['train', 'test', 'val'], default='test',
        help='define which split to run inference on')

    return parser.parse_args()

if __name__ == '__main__':
    args = _parse_args()
    main(
      config_path=args.config,
      checkpoint=args.checkpoint,
      split=args.split,
    )
