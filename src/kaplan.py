import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import gc
import shutil
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

import sys
sys.path.append('utils')

from surv_utils import generate_mapping, stack_tensors, create_bins, plot_box
from kaplan_utils import train_test, dataHum
from seeds import seed_torch
from losses_dataset import SurvivalDataset

parser = argparse.ArgumentParser(description='Survival Analysis')
parser.add_argument('--outcome', type=str, default='Outcome at last FU', help='The column name for the outcome')
parser.add_argument('--logs', type=str, default='logs', help='The folder for the Tensorboard results')
parser.add_argument('--trials', type=int, default=0, required=True, help='The trials that we are trying')
parser.add_argument('--epochs', type=int, default=100, help='The number of epochs for training')
parser.add_argument('--cuda', type=bool, default=True, help='Set to True if you want to use GPU')
parser.add_argument('--features', type=str, default='features_mgg', help='Folder for the features')
parser.add_argument('--pca', type=bool, default=False, help='Use PCA for the features')
parser.add_argument('--complete', type=bool, default=False)


def create_loader(args):
    if not args.cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    print('Device:', device)

    seed_torch(seed=42, device=device)

    mapping = generate_mapping(args)
    modifed_data = create_bins(data=dataHum, label_col=None, n_bins=4, eps=1e-6)

    if not os.path.exists(os.path.join(args.logs, str(args.trials))):
        os.makedirs(os.path.join(args.logs, str(args.trials)), exist_ok=True)
    else:
        shutil.rmtree(os.path.join(args.logs, str(args.trials)))
        os.makedirs(os.path.join(args.logs, str(args.trials)), exist_ok=True)

    if not os.path.exists(os.path.join('results_final', str(args.trials))):
        os.makedirs(os.path.join('results_final', str(args.trials)), exist_ok=True)
    else:
        shutil.rmtree(os.path.join('results_final', str(args.trials)))
        os.makedirs(os.path.join('results_final', str(args.trials)), exist_ok=True)

    for cv, split in enumerate(os.listdir('splits')):
        if not os.path.exists(os.path.join(args.logs, str(args.trials), f'cv_{cv}')):
            os.mkdir(os.path.join(args.logs, str(args.trials), f'cv_{cv}'))

        writer = SummaryWriter(os.path.join(args.logs, str(args.trials), f'cv_{cv}'), flush_secs=15)

        if split.endswith('.csv'):
            print(f'\nCross Validation: Fold {cv}')
            split_data = pd.read_csv(f'splits/{split}')
            ids_train = split_data['train']
            ids_val = split_data['val']

            if ids_val.dtype == 'float64':
                ids_val = ids_val.astype('Int64')

            if ids_train.dtype == 'float64':
                ids_train = ids_train.astype('Int64')

            outcome_train = [1 if dataHum.loc[str(id), args.outcome] == 'Dead' else 0 for id in ids_train if str(id) in mapping.keys()]
            outcome_val = [1 if dataHum.loc[str(id), args.outcome] == 'Dead' else 0 for id in ids_val if str(id) in mapping.keys()]
            
            features_train = {str(id): stack_tensors(args, mapping[str(id)]) for id in ids_train if str(id) in mapping.keys() and str(id).lower() != "nan"}
            features_val = {str(id): stack_tensors(args, mapping[str(id)]) for id in ids_val if str(id) in mapping.keys() and str(id).lower() != "nan"}

            times_train = [modifed_data.loc[str(id), 'time_label'] for id in ids_train if str(id) in mapping.keys()]
            times_val = [modifed_data.loc[str(id), 'time_label'] for id in ids_val if str(id) in mapping.keys()]

            real_times_train = [modifed_data.loc[str(id), 'OS censored at TPX  months'] for id in ids_train if str(id) in mapping.keys()]
            real_times_val = [modifed_data.loc[str(id), 'OS censored at TPX  months'] for id in ids_val if str(id) in mapping.keys()]
    
            train_dataset = SurvivalDataset(list(features_train.keys()), list(features_train.values()), np.array(times_train), np.array(real_times_train), np.array(outcome_train))
            val_dataset = SurvivalDataset(list(features_val.keys()), list(features_val.values()), np.array(times_val), np.array(real_times_val), np.array(outcome_val))

            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            train_test(args, train_loader, val_loader, writer, cv, device)
    
    plot_box(args)


if __name__ == '__main__':
    args = parser.parse_args()
    create_loader(args)