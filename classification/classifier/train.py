'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
'''

import os
import time
import argparse
import yaml
import glob
from tqdm import trange
from pathlib import Path
import json

from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np

# let's import our own classes and functions!
from util import init_seed
from dataset import create_dataloader
from model import CustomResNet18

from dotenv import dotenv_values

# init comet experiement
env_vars = {key: val for key, val in dotenv_values('./classifier/.env').items()}
experiment = Experiment(
  api_key=env_vars['COMET_API_KEY'],
  project_name=env_vars['COMET_PROJECT_NAME'],
  workspace=env_vars['COMET_WORKSPACE']
)

def load_model(cfg, resume):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    checkpoints_dir = os.path.join(cfg['experiment_dir'], 'checkpoints')
    model_states = glob.glob(checkpoints_dir + '/*.pt')

    if len(model_states) and resume == True:
        # at least one save state found; get latest
        model_epochs = [int(m.replace(checkpoints_dir,'').replace('/','').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(f'{checkpoints_dir}/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found and/or resume flag was set to false; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch



def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    checkpoints_dir = os.path.join(cfg['experiment_dir'], 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(f'{checkpoints_dir}/{epoch}.pt', 'wb'))
    
    # also save config file if not present
    yaml_files = glob.glob('*.yaml') + glob.glob('*.yml')
    if len(yaml_files) == 0:
        cfpath = f'{checkpoints_dir}/config.yaml'
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)


def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer



def train(cfg, dataLoader, model, optimizer):
    '''
        Our actual training function.
    '''
    device = cfg['device']

    # put model on device
    model.to(device)
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # loss function
    # NOTE: this will return the MEAN loss for all images in the batch, see
    # default value of "reduction" param in docs:
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    criterion = nn.CrossEntropyLoss()

    # running averages
    loss_total, oa_total = 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)

    # time profiling
    # current_time = time.time
    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    for idx, (data, labels, image_path) in enumerate(dataLoader):       # see the last line of file "dataset.py" where we return the image tensor (data) and label

        # put data and labels on device
        data, labels = data.to(device), labels.to(device)

        # forward pass
        prediction = model(data)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor

        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total



def validate(cfg, dataLoader, model, categories):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)
    all_labels, all_preds = [], []
    # all_images = []

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels, image_path) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            preds = torch.argmax(prediction, dim=1)
            oa = torch.mean((preds == labels).float())
            oa_total += oa.item()
            # all_images.append(data)
            all_labels = all_labels + labels.tolist()
            all_preds = all_preds + preds.tolist()

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    # calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    # print(f'confusion matrix:')
    # print(cm)

    true_pos = np.diag(cm)
    false_pos = np.sum(cm, axis=0) - true_pos
    false_neg = np.sum(cm, axis=1) - true_pos
    # print(f'true_pos: {true_pos}')
    # print(f'false_pos: {false_pos}')
    # print(f'false_neg: {false_neg}')

    # find weighted avg precision & recall of all classes
    # see docs for average param: 
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    avg_precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    avg_recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    # print(f'scikit_precision - weighted (avg for all classes): {avg_precision_weighted}')
    # print(f'scikit_recall - weighted (avg for all classes): {avg_recall_weighted}')

    # find precssion and recall for each class
    precision_by_class = true_pos / (true_pos + false_pos)
    recall_by_class = true_pos / (true_pos + false_neg)
    # print(f'precision (for each class): {precision_by_class}')
    # print(f'recall (for each class): {recall_by_class}')
    category_stats = {}
    for i, class_p in enumerate(precision_by_class):
        metric_name = categories[i]['name'].replace(' ', '_') + '_precision_val'
        category_stats[metric_name] = class_p
    for i, class_r in enumerate(recall_by_class):
        metric_name = categories[i]['name'].replace(' ', '_') + '_recall_val'
        category_stats[metric_name] = class_r

    return {
      'loss': loss_total, 
      'oa': oa_total,
      'cm': cm,
      'avg_precision_weighted': avg_precision_weighted,
      'avg_recall_weighted': avg_recall_weighted,
      'category_stats': category_stats,
      # TODO: passing images to log_confusion_matrix wasn't working as expected
      # look into using confusion matrix callback to acomplish this like:
      # https://colab.research.google.com/github/comet-ml/comet-examples/blob/master/notebooks/Comet-Confusion-Matrix.ipynb#scrollTo=CDjKMs40CiG-
      # 'images': all_images
    }


def main(config_path, resume):

    # load config, send hyper params to comet
    print(f'Using config "{config_path}"')
    experiment_dir = Path(config_path).parent.absolute()
    cfg = yaml.safe_load(open(config_path, 'r'))
    cfg['experiment_dir'] = experiment_dir
    hyper_params = {
      'num_epochs': cfg.get('num_epochs', None),
      'batch_size': cfg.get('batch_size', None),
      'learning_rate': cfg.get('learning_rate', None),
      'weight_decay': cfg.get('weight_decay', None),
    }
    experiment.log_parameters(hyper_params)

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'


    # load annotation file (# TODO: we also do this in dataset.py. Make more DRY)
    annoPath = os.path.join(cfg['data_root'], 'train_cct.json')
    meta = json.load(open(annoPath, 'r'))
    categories = meta['categories']
    cm_labels = [c['name'] for c in categories]

    # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, split='train')
    dl_val = create_dataloader(cfg, split='val')

    # initialize model
    model, current_epoch = load_model(cfg, resume)

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    # start_epoch = current_epoch # just for testing 1 epoch
    # numEpochs = start_epoch + 1 # just for testing 1 epoch
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train = train(cfg, dl_train, model, optim)
        val_metrics = validate(cfg, dl_val, model, categories)
        
        stats = {
            'loss_train': loss_train,
            'loss_val': val_metrics['loss'],
            'oa_train': oa_train,
            'oa_val': val_metrics['oa'],
            'avg_precision_weighted_val': val_metrics['avg_precision_weighted'],
            'avg_recall_weighted_val': val_metrics['avg_recall_weighted']
        }
        # log to comet
        experiment.log_metrics(stats, step=current_epoch, epoch=current_epoch)
        experiment.log_metrics(val_metrics['category_stats'], step=current_epoch, epoch=current_epoch)

        cm_labels = [c['name'] for c in categories]
        experiment.log_confusion_matrix(matrix=val_metrics['cm'], labels=cm_labels,
          file_name="confusion-matrix-epoch-{}.json".format(current_epoch),
          row_label="Actual Category",
          column_label="Predicted Category", epoch=current_epoch)

        # save model locally
        save_model(cfg, current_epoch, model, stats)
  
    # That's all, folks!
        

def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Train deep learning model.')

    parser.add_argument(
      '--config', 
      default='configs/exp_resnet18.yaml',
      help='Path to config file')

    parser.add_argument(
        '--resume', action='store_true',
        help='include --resume flag to resume training from an existing '
        'checkpoint')
    
    parser.add_argument(
        '--no-resume', dest='resume', action='store_false',
        help='include --no-resume flag to start training from scratch ')
    
    parser.set_defaults(resume=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = _parse_args()
    main(config_path=args.config,
        resume=args.resume)
