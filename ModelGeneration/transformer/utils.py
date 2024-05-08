import numpy as np
import torch
import sys
import builtins
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from copy import deepcopy


def Initialization(config):
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))
    return device

def Setup(args):
    """
        Input:
            args: arguments object from argparse
        Returns:
            config: configuration dictionary
    """
    config = args.__dict__  # configuration dictionary
    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    model = config['xfile'].split("\\")[-1]
    model = model[2:-4]
    output_dir = os.path.join(output_dir,model, initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    #config['pred_dir'] = os.path.join(output_dir, 'predictions')
    #config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    #create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])
    create_dirs([config['save_dir']])
    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

  

    return config

def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

     
class myDataLoader(torch.utils.data.Dataset):
  def __init__(self, X_path, y_path, output,test_size=0.2, val_size=0.1, random_state=42, normalize=False):
    self.X = np.load(X_path)
    self.y = np.load(y_path)
    torque_indices = np.arange(0,7,1)
    position_error_indices = np.arange(28,35,1)
    velocity_error_indices = np.arange(35,42,1)
    selection = np.concatenate((torque_indices, position_error_indices, velocity_error_indices))
    self.X = self.X[:, :, selection]
    label_encoder = LabelEncoder()
    self.y = label_encoder.fit_transform(self.y)
    self.X = np.swapaxes(self.X, 1, 2) # swap axes such that #samples, #features #winwdowlength

    # Split data into train/test sets
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
    mean = []
    std = []
    if normalize:
        for i in range(self.X_train.shape[1]):
            scaler = StandardScaler()
            self.X_train[:, i, :] = scaler.fit_transform(self.X_train[:, i,:]) 
            self.X_test[:, i, :] = scaler.transform(self.X_test[:, i, :])
            mean.append((scaler.mean_).tolist()) 
            std.append((scaler.scale_).tolist())
        data  = {'normalization_mean':mean,
                 'normalization_std': std}
        with open(os.path.join(output, 'configuration.json'), 'r') as f:
            config = json.load(f)
        config.update(data)
        with open(os.path.join(output, 'configuration.json'), 'w') as f:
            json.dump(config, f)
             

    # Further split train data into train/val sets
    if val_size > 0:
      self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=val_size, random_state=random_state)
    else:
      self.X_val = None
      self.y_val = None
    

  def __len__(self):
    # Depending on usage, return length of appropriate data
    if hasattr(self, 'X_train'):
      return len(self.X_train)
    elif hasattr(self, 'X_val'):
      return len(self.X_val)  # Access validation data length
    else:
      return len(self.X_test)

  def __getitem__(self, idx):
    if hasattr(self, 'X_train'):  # Access training data
      return torch.tensor(self.X_train[idx],dtype=torch.float32), torch.tensor(self.y_train[idx],dtype=torch.long),idx
    elif hasattr(self, 'X_val'):  # Access validation data
      return torch.tensor(self.X_val[idx],dtype=torch.float32), torch.tensor(self.y_val[idx],dtype=torch.long),idx
    else:  # Access test data
      return torch.tensor(self.X_test[idx],dtype=torch.float32), torch.tensor(self.y_test[idx],dtype=torch.long),idx


class Printer(object):
    """Class for printing output by refreshing the same line in the console, e.g. for indicating progress of a process"""

    def __init__(self, console=True):

        if console:
            self.print = self.dyn_print
        else:
            self.print = builtins.print

    @staticmethod
    def dyn_print(data):
        """Print things to stdout on one line, refreshing it dynamically"""
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


def readable_time(time_difference):
    """Convert a float measuring time difference in seconds into a tuple of (hours, minutes, seconds)"""

    hours = time_difference // 3600
    minutes = (time_difference // 60) % 60
    seconds = time_difference % 60

    return hours, minutes, seconds

import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
# plt.style.use('ggplot')


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time} secs")
        return value
    return wrapper_timer


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, path):

        if current_valid_loss < self.best_valid_loss:

            self.best_valid_loss = current_valid_loss
            print(f"Best validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch}\n")
            save_model(path, epoch, model, optimizer)


def load_model(model, model_path, optimizer=None, resume=False, change_output=False,
               lr=None, lr_step=None, lr_factor=None):
    
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = deepcopy(checkpoint['state_dict'])
    if change_output:
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('output_layer'):
                state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    print('Loaded model from {}. Epoch: {}'.format(model_path, checkpoint['epoch']))
    start_epoch = checkpoint['epoch']
    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for i in range(len(lr_step)):
                if start_epoch >= lr_step[i]:
                    start_lr *= lr_factor[i]
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model