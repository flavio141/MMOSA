import os
import gc
import pickle
import optuna
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import warnings
sys.path.append('models')
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

import sys
sys.path.append('utils')

from seeds import seed_torch
from surv_utils import create_bins, write_to_csv
from sksurv.metrics import concordance_index_censored


from models import MultiOmicsDF
from torch.utils.data import DataLoader
from losses_dataset import MultiomicsDataset, cox_ph_loss


os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "2"


dataHum = pd.read_csv('dataset/MultiomicsFinal.csv', index_col='ID')

parser = argparse.ArgumentParser(description='Survival Analysis similar to MOMA')
parser.add_argument('--outcome', type=str, default='Outcome at last FU', help='The column name for the outcome')
parser.add_argument('--epochs', type=int, default=30, help='The number of epochs for training')
parser.add_argument('--cuda', type=bool, default=True, help='Set to True if you want to use GPU')
parser.add_argument('--features', type=str, default='features_giga', help='Folder for the features')


def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def write_information(path, model, optimizer, criterion, epochs):
    with open(path, "w") as file:
        file.write("Model Information\n\n")
        file.write(f"Epochs --> {epochs}\n")
        file.write(f"Model Architecture --> {model}\n")
        file.write(f"Optimizer --> {optimizer}\n")
        file.write(f"Loss Function --> {criterion}\n")
        file.write("\n")


def plot_box(trial):
    c_indexes = []

    for file in os.listdir(os.path.join('optuna', 'MultiOmics')):
        if f'trial{str(trial)}' in file and file.endswith('.csv'):
            c_indexes.append(('_'.join(['Fold', file.split('.')[0].split('_')[-1]]), pd.read_csv(os.path.join('optuna', 'MultiOmics', file))['Val_C-Index']))

    folds = [item[0] for item in c_indexes]
    c_values = [list(item[1]) for item in c_indexes]
    cmap = plt.get_cmap('tab10', len(folds))

    plt.figure(figsize=(10, 6))
    box = plt.boxplot(c_values, patch_artist=True)

    for patch, color in zip(box['boxes'], cmap.colors):
        patch.set_facecolor(color)

    plt.xticks(range(1, len(folds) + 1), folds)
    plt.xlabel('Fold')
    plt.ylabel('C-Index')
    plt.title('Boxplot of for each Fold')
    plt.show()
    plt.savefig(f'optuna/MultiOmics/trial{str(trial)}_boxplot.png')


def train_test_optuna(args, train_loader, val_loader, parameters, trial_info, trial, cv, device):
    
    mode = 'w'
    filename = f"trial{str(trial)}_results_{cv}.csv"

    csv_folder = os.path.join('optuna', 'MultiOmics')

    model = MultiOmicsDF(input_size=1536, dropout=0.4)
    model.to(device)

    optimizers = {
        'Adam': optim.Adam(model.parameters(), lr=1e-04, weight_decay=1e-03),
        'AW': optim.AdamW(model.parameters(), lr=1e-05, weight_decay=1e-05),
        'ASGD': optim.ASGD(model.parameters(), lr=1e-05, weight_decay=1e-04),
        'RMS': optim.RMSprop(model.parameters(), lr=1e-04, weight_decay=1e-05, momentum=0.5),
        'SGD': optim.SGD(model.parameters(), lr=1e-05, weight_decay=1e-04, momentum=0.9, nesterov=True)
    }

    criterion_cox = cox_ph_loss
    optimizer = optimizers[parameters['optimizer_name']]

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min = 0, last_epoch = -1)

    seed_torch(seed=42, device=device)
    write_information(os.path.join(csv_folder, f'trial{str(trial)}_information.txt'), model, optimizer, criterion_cox, parameters['epochs'])

    c_indexes = {}

    model.train()
    for epoch in range(parameters['epochs']):
        total_loss = 0

        all_risk_scores = []
        all_censorships = []
        all_real_times = []


        for train_sample in train_loader:
            features, demog, clin, genomic, transcr, real_time, event_indicator, masking = train_sample['staining'], train_sample['demog'], train_sample['clin'], train_sample['genomic'], train_sample['transcr'], train_sample['time'], train_sample['outcome'], train_sample['masking']
            features, demog, clin, genomic, transcr, real_time, event_indicator, masking = features.to(device), demog.to(device), clin.to(device), genomic.to(device), transcr.to(device), real_time.to(device), event_indicator.to(device), masking.to(device)
            optimizer.zero_grad()

            risk = model(features, masking, demog, clin, genomic, transcr)
            loss = criterion_cox(risk, real_time, event_indicator) 

            all_risk_scores.append(risk.detach().cpu().numpy())
            all_censorships.append(event_indicator.detach().cpu().numpy())
            all_real_times.append(real_time.detach().cpu().numpy())

            del risk, event_indicator, real_time, features, masking, demog, clin, genomic, transcr

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=10, norm_type=2.0)
            torch.cuda.empty_cache()

        all_risk_scores = np.concatenate(all_risk_scores)
        all_censorships = np.concatenate(all_censorships)
        all_real_times = np.concatenate(all_real_times)

        valid_indices = ~np.isnan(all_censorships) & ~np.isnan(all_real_times) & ~np.isnan(all_risk_scores)
        valid_indices = valid_indices[:all_censorships.shape[0]]

        filtered_censorships = all_censorships[valid_indices]
        filtered_real_times = all_real_times[valid_indices]
        filtered_risk_scores = all_risk_scores[valid_indices]

        del all_censorships, all_real_times, all_risk_scores

        if not (np.isnan(filtered_censorships).any() or np.isnan(filtered_real_times).any() or np.isnan(filtered_risk_scores).any()):
            c_index = concordance_index_censored(
                (filtered_censorships.reshape(-1,)).astype(bool), 
                filtered_real_times, 
                filtered_risk_scores.reshape(-1,), 
                tied_tol=1e-08
            )[0]
        else:
            c_index = 0.5
        
        scheduler.step(total_loss)
        model.eval()

        with torch.no_grad():
            eval_loss = 0
            eval_risk_scores = []
            eval_censorships = []
            eval_real_times = []

            for val_sample in val_loader:
                features_val, demog_val, clin_val, genomic_val, transcr_val, real_times_val, event_indicator_val, masking_val = val_sample['staining'], val_sample['demog'], val_sample['clin'], val_sample['genomic'], val_sample['transcr'], val_sample['time'], val_sample['outcome'], val_sample['masking']
                #features_val = F.normalize(features_val, p=2, dim=0)
                features_val, demog_val, clin_val, genomic_val, transcr_val, real_times_val, event_indicator_val, masking_val = features_val.to(device), demog_val.to(device), clin_val.to(device), genomic_val.to(device), transcr_val.to(device), real_times_val.to(device), event_indicator_val.to(device), masking_val.to(device)

                eval_risk = model(features_val, masking_val, demog_val, clin_val, genomic_val, transcr_val)
                loss_cox_test = criterion_cox(eval_risk, real_times_val, event_indicator_val) 
                eval_loss += loss_cox_test

                eval_risk_scores.append(eval_risk.detach().cpu().numpy())
                eval_censorships.append(event_indicator_val.detach().cpu().numpy())
                eval_real_times.append(real_times_val.detach().cpu().numpy())

                del eval_risk, event_indicator_val, real_times_val, features_val, masking_val, demog_val, clin_val, genomic_val, transcr_val

                torch.cuda.empty_cache()

            eval_final = eval_loss /len(val_loader)

            eval_risk_scores = np.concatenate(eval_risk_scores)
            eval_censorships = np.concatenate(eval_censorships)
            eval_real_times = np.concatenate(eval_real_times)

            valid_indices_val = ~np.isnan(eval_real_times) & ~np.isnan(eval_censorships) & ~np.isnan(eval_risk_scores)
            valid_indices_val = valid_indices_val[:eval_censorships.shape[0]]

            filtered_censorships_val = eval_censorships[valid_indices_val]
            filtered_real_times_val = eval_real_times[valid_indices_val]
            filtered_risk_scores_val = eval_risk_scores[valid_indices_val]

            del eval_risk_scores, eval_censorships, eval_real_times

            if not (np.isnan(filtered_censorships_val).any() or np.isnan(filtered_real_times_val).any() or np.isnan(filtered_risk_scores_val).any()):
                c_index_val = concordance_index_censored(
                    (filtered_censorships_val.reshape(-1,)).astype(bool), 
                    filtered_real_times_val, 
                    filtered_risk_scores_val.reshape(-1,), 
                    tied_tol=1e-08
                )[0]
            else:
                c_index_val = 0.5
                
            c_indexes[epoch] = c_index_val

        if (epoch + 1) % 5 == 0:
            print('Epoch: {}, Train_loss: {:.4f}, Val_Loss: {:.4f}, Train_c_index: {:.4f}, Val_c_index: {:.4f}'.format(epoch + 1, total_loss, eval_final.item(), c_index, c_index_val))
            trial_info.report(c_index_val, trial)

            if trial_info.should_prune():
                raise optuna.exceptions.TrialPruned()
        # if (epoch + 1) % 10 == 0:
        #     print('Epoch: {}, Train_loss: {:.4f}, Val_Loss: {:.4f}, Train_c_index: {:.4f}, Val_c_index: {:.4f}'.format(epoch + 1, total_loss, eval_final.item(), c_index, c_index_val))

        data = [epoch, total_loss, eval_final.item(), c_index, c_index_val, cv]
        write_to_csv(os.path.join(csv_folder, filename), data, mode)

        mode = 'a'

    return np.mean(list(c_indexes.values())[-3:])


def optuna_loader(args, parameters, trial_info, trial):
    if not args.cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    
    if device.type == 'cuda:1':
        torch.cuda.empty_cache()
        gc.collect()
    print('Device:', device)

    seed_torch(seed=42, device=device)

    modifed_data = create_bins(data=dataHum, label_col=None, n_bins=4, eps=1e-6)

    dataset_dict = {}
    for file in os.listdir('dataset_multiomics'):
        if file.endswith('.pickle'):
            with open(f"dataset_multiomics/{file}", 'rb') as f:
                pickle_data = pickle.load(f)
            
            for key in pickle_data.keys():
                if key not in dataset_dict.keys():
                    dataset_dict[key] = pickle_data[key]
                    dataset_dict[key]['time'] = modifed_data.loc[key, 'OS censored at TPX  months']


    if not os.path.exists(os.path.join('optuna', 'MultiOmics')):
        os.makedirs(os.path.join('optuna', 'MultiOmics'), exist_ok=True)

    c_indexes = {}

    for cv, split in enumerate(os.listdir('splits')):

        if split.endswith('.csv'):
            print(f'\nCross Validation: Fold {cv}')
            split_data = pd.read_csv(f'splits/{split}')
            ids_train = split_data['train']
            ids_val = split_data['val']

            if ids_val.dtype == 'float64':
                ids_val = ids_val.astype('Int64')

            if ids_train.dtype == 'float64':
                ids_train = ids_train.astype('Int64')

            train_dict = {patient_id: dataset_dict[patient_id] for patient_id in ids_train if patient_id in dataset_dict and '_dp' not in patient_id}
            val_dict = {patient_id: dataset_dict[patient_id] for patient_id in ids_val if patient_id in dataset_dict and '_dp' not in patient_id}

            train_dataset = MultiomicsDataset(train_dict, rows=15000)
            val_dataset = MultiomicsDataset(val_dict, rows=15000)

            train_loader = DataLoader(train_dataset, batch_size=10, collate_fn=custom_collate_fn, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=5, collate_fn=custom_collate_fn, shuffle=False)

            c_indexes[cv] = train_test_optuna(args, train_loader, val_loader, parameters, trial_info, trial, cv, device)
    
    plot_box(trial)
    return c_indexes


def objective(trial):
    lr = trial.suggest_categorical('lr', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-6, 1e-5, 1e-4, 1e-2])

    optimizer_name = trial.suggest_categorical('optimizer_name', ['Adam', 'RMS', 'SGD'])
    dropout = trial.suggest_uniform('dropout', 0.3, 0.7)
    epochs = trial.suggest_int('epochs', 10, 30)

    args = parser.parse_args()
    parameters = {
        'lr': lr,
        'weight_decay': weight_decay,
        'optimizer_name': optimizer_name,
        'dropout': dropout,
        'epochs': epochs
    }

    c_indexes = optuna_loader(args, parameters, trial, trial._trial_id)
    return np.mean(list(c_indexes.values()))


if __name__ == '__main__':
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    storage = "sqlite:///optuna_study.db"
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        pruner=pruner,
        study_name='OptunaMultiomics',
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open("optuna_results.txt", "w") as f:
        f.write("Number of finished trials: {}\n".format(len(study.trials)))

        f.write("Best trial:\n")
        f.write("  Value: {}\n".format(trial.value))
        f.write("  Params:\n")
        for key, value in trial.params.items():
            f.write("    {}: {}\n".format(key, value))

    print("Results in optuna_results.txt")