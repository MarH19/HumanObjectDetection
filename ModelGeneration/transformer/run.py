import os
import argparse
import logging
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Import Project Modules -----------------------------------------------------------------------------------------------
from utils import Setup, myDataLoader, Initialization,load_model
from model import model_factory, count_parameters
from optimizers import get_optimizer
from loss import get_loss_module
from Training import SupervisedTrainer, train_runner
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
logger = logging.getLogger('__main__')


def choose_dataset():
    processed_data_path = Path(os.environ.get(
        "DATASET_REPO_ROOT_PATH")) / "processedData"

    sub_repo = dict([(str(i), p) for i, p in enumerate(
        processed_data_path.iterdir()) if p.is_dir()])
    print("sub repo:")
    lines = [f'{key} {value.name}' for key, value in sub_repo.items()]
    print('\n'.join(lines) + '\n')
    subrepo_key = None
    while subrepo_key not in sub_repo:
        subrepo_key = input(
            "Which sub repo should be used? (choose by index): ")

    full_path = processed_data_path / sub_repo[subrepo_key]

    datasets = dict([(str(i), p) for i, p in enumerate(full_path.iterdir())
                     if p.is_file and p.name.startswith("x_") and p.suffix == ".npy" and "test" not in p.name])
    lines = [f'{key} {value.name}' for key, value in datasets.items()]

    print("Datasets:")
    print('\n'.join(lines) + '\n')
    dataset_key = None
    while dataset_key not in datasets:
        dataset_key = input(
            "Which dataset should be used? (choose by index): ")
    return sub_repo[subrepo_key], datasets[dataset_key]
def choose_normalization_mode():
    normalization_choice = ""
    while normalization_choice not in ["y", "n"]:
        normalization_choice = input(
            "Should the data be normalized? (y / n): ").lower()
    return True if normalization_choice == "y" else False


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    sub_repo,xfile = choose_dataset()
    normalize = choose_normalization_mode()
    yfile = (xfile.parent / xfile.name.replace("x_", "y_")).absolute()
    output_dir = Path(os.environ.get("TRANSFORMER_RESULT_PATH"))
    config = {  'output_dir':output_dir.absolute(),
                'xfile':xfile,
                'yfile':yfile,
                'Norm':normalize,
                'val_ratio':0.1,
                'Net_Type':['C-T'], #choices={'T', 'C-T'}, help="Network Architecture. Convolution (C)" "Transformers (T)"
                'emb_size':16,
                'dim_ff':256,
                'num_heads':8,
                'Fix_pos_encode':'tAPE', #{'tAPE', 'Learn', 'None'}
                'Rel_pos_encode':'eRPE', #{'eRPE', 'Vector', 'None'}
                'epochs':100,
                'batch_size':512,
                'lr':1e-3,
                'dropout':0.2,
                'val_interval':2,
                'key_metric':'loss', #{'loss', 'accuracy', 'precision'}, help='Metric used for defining best epoch'
                'l2reg':0.01,
                'gpu':'0',
                'seed':1234,
                'console':False #"Optimize printout for console output; otherwise for file"
    }


    config = Setup(config)  # configuration dictionary
    device = Initialization(config)
    # ------------------------------------ Load Data ---------------------------------------------------------------
    logger.info("Loading Data ...")
    dataset = myDataLoader(config['xfile'],config['yfile'],output=config['output_dir'],test_size=0.2, val_size=0.1,normalize=config['Norm'])
    train_loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    test_loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    # --------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Build Model -----------------------------------------------------
    logger.info("Creating model ...")
    config['Data_shape'] = dataset.X_train.shape
    config['num_labels'] = np.unique(dataset.y_train).shape[0]
    model = model_factory(config)
    logger.info("Total number of parameters: {}".format(count_parameters(model)))
    # -------------------------------------------- Model Initialization ------------------------------------
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], decoupled_weight_decay=True, weight_decay=config['l2reg']) # decoupled.. for having RadamW
    config['loss_module'] = get_loss_module()
    save_path = os.path.join(config['save_dir'],'model_{}.pth'.format('last'))
    tensorboard_writer = SummaryWriter('summary')
    model.to(device)
    # ---------------------------------------------- Training The Model ------------------------------------
    logger.info('Starting training...')
    trainer = SupervisedTrainer(model, train_loader, device, config['loss_module'], config['optimizer'], console=config['console'], print_conf_mat=False)
    val_evaluator = SupervisedTrainer(model, val_loader, device, config['loss_module'], console=config['console'],
                                    print_conf_mat=False)

    train_runner(config, model, trainer, val_evaluator, save_path)
    best_model, optimizer, start_epoch = load_model(model, save_path, config['optimizer'])
    best_model.to(device)
    best_test_evaluator = SupervisedTrainer(best_model, test_loader, device, config['loss_module'], console=config['console'],
                                            print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True,epoch_num=start_epoch)
    print_str = 'Best Model Test Summary: '
    for k, v in best_aggr_metrics_test.items():
        print_str += '{}: {} | '.format(k, v)
    print(print_str)
