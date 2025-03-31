import os
import gc
import shutil
import numpy as np
import pandas as pd
import argparse

from sksurv.metrics import concordance_index_censored

from tensorboardX import SummaryWriter

import pickle
import sys
import warnings
sys.path.append('models')
sys.path.append('utils')
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from seeds import seed_torch
from surv_utils import generate_mapping, stack_tensors, create_bins, plot_box, write_to_csv, write_information

from torch.utils.data import DataLoader
from models import ModelSurv
from focal_loss.focal_loss import FocalLoss
from losses_dataset import NLLSurvLoss, CensoredCrossEntropyLoss, CoxLoss, WeibulLoss, SurvivalDataset, WeibulLossContinuous, cox_ph_loss, LogCoshLoss

os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "2"


dataHum = pd.read_csv('dataset/MultiomicsFinal.csv', index_col='ID')
num_features = 1536

parser = argparse.ArgumentParser(description='Survival Analysis similar to MOMA')
parser.add_argument('--outcome', type=str, default='Outcome at last FU', help='The column name for the outcome')
parser.add_argument('--logs', type=str, default='logs', help='The folder for the Tensorboard results')
parser.add_argument('--trials', type=int, default=100, required=False, help='The trials that we are trying')
parser.add_argument('--epochs', type=int, default=30, help='The number of epochs for training')
parser.add_argument('--cuda', type=bool, default=True, help='Set to True if you want to use GPU')
parser.add_argument('--features', type=str, default='features_dino', help='Folder for the features')
parser.add_argument('--complete', type=bool, default=False)


def create_masking(features_train):
    masking = {}
    features_copy = {}
    to_remove = []

    max_rows = max(feature.shape[0] for feature in features_train.values())

    for idx, (patient, feature) in enumerate(features_train.items()):
        if feature.shape[0] > max_rows:#10000:
            to_remove.append(idx)
        else:
            features_padding = torch.full((max_rows, num_features), -999, dtype=feature.dtype)
            features_padding[:feature.shape[0], :] = feature
            
            features_copy[patient] = features_padding
            masking[patient] = torch.tensor(feature.shape[0])
    return features_copy, masking, to_remove



def train_test_moma(args, train_loader, val_loader, writer, cv, device):
    mode = 'w'
    filename = f"results_{cv}.csv"

    csv_folder = os.path.join('results_final', str(args.trials))

    model = ModelSurv(input_size=num_features, surv_nodes=[num_features, 64, 1], dropout=0.4)
    model.to(device)

    criterions = {
        'CE' : nn.CrossEntropyLoss(reduction='mean'),
        'NLL' : NLLSurvLoss(alpha=0.8),
        'Focal' : FocalLoss(gamma=4),
        'Censored' : CensoredCrossEntropyLoss(),
        'Cox' : CoxLoss(),
        'WB' : WeibulLoss(),
        'WBC': WeibulLossContinuous(),
        'VAE': LogCoshLoss(),
        'VAECox': cox_ph_loss
    }

    optimizers = {
        'AD': optim.Adam(model.parameters(), lr=1e-05, weight_decay=1e-05),
        'AW': optim.AdamW(model.parameters(), lr=1e-05, weight_decay=1e-05),
        'ASGD': optim.ASGD(model.parameters(), lr=1e-05, weight_decay=1e-04),
        'RMS': optim.RMSprop(model.parameters(), lr=1e-04, weight_decay=1e-05, momentum=0.5),
        'SGD': optim.SGD(model.parameters(), lr=1e-05, weight_decay=1e-04, momentum=0.9, nesterov=True)
    }

    criterion_cox = criterions['VAECox']
    optimizer = optimizers['AD']

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 0, last_epoch = -1)

    seed_torch(seed=42, device=device)
    write_information(args, os.path.join(csv_folder, 'trial_information.txt'), model, optimizer, criterion_cox, args.epochs, scheduler)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0

        all_risk_scores = []
        all_censorships = []
        all_event_times = []
        all_real_times = []


        for _, features, time_indicator, real_time, event_indicator, masking in train_loader:
            if 1 not in event_indicator:
                continue
            features, time_indicator, real_time, event_indicator, masking = features.to(device), time_indicator.to(device), real_time.to(device), event_indicator.to(device), masking.to(device)

            optimizer.zero_grad()
            risk = model(features, masking)
            loss = criterion_cox(risk, real_time, event_indicator) 

            all_risk_scores.append(risk.detach().cpu().numpy())
            all_censorships.append(event_indicator.detach().cpu().numpy())
            all_event_times.append(time_indicator.detach().cpu().numpy())
            all_real_times.append(real_time.detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            del features, time_indicator, real_time, event_indicator, masking

            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2.0)
            torch.cuda.empty_cache()

        try:
            all_risk_scores = np.concatenate(all_risk_scores)
            all_censorships = np.concatenate(all_censorships)
            all_event_times = np.concatenate(all_event_times)
            all_real_times = np.concatenate(all_real_times)

            c_index = concordance_index_censored((all_censorships).astype(bool), all_real_times, all_risk_scores.reshape(-1,), tied_tol=1e-08)[0]
        except Exception as e:
            print(e)
            c_index = 0.5
        scheduler.step(total_loss)

        model.eval()
        with torch.no_grad():
            eval_loss = 0
            eval_risk_scores = []
            eval_censorships = []
            eval_event_times = []
            eval_real_times = []

            for _, features_val, time_indicator_val, real_times_val, event_indicator_val, masking_val in val_loader:
                if 1 not in event_indicator:
                    continue
                features_val, time_indicator_val, real_times_val, event_indicator_val, masking_val = features_val.to(device), time_indicator_val.to(device), real_times_val.to(device), event_indicator_val.to(device), masking_val.to(device)

                eval_risk = model(features_val, masking_val)
                loss_cox_test = criterion_cox(eval_risk, real_times_val, event_indicator_val) 
                eval_loss += loss_cox_test

                eval_risk_scores.append(eval_risk.detach().cpu().numpy())
                eval_censorships.append(event_indicator_val.detach().cpu().numpy())
                eval_event_times.append(time_indicator_val.detach().cpu().numpy())
                eval_real_times.append(real_times_val.detach().cpu().numpy())

                torch.cuda.empty_cache()

            eval_final = eval_loss /len(val_loader)

            try:
                eval_risk_scores = np.concatenate(eval_risk_scores)
                eval_censorships = np.concatenate(eval_censorships)
                eval_event_times = np.concatenate(eval_event_times)
                eval_real_times = np.concatenate(eval_real_times)

                c_index_val = concordance_index_censored((eval_censorships).astype(bool), eval_real_times, eval_risk_scores.reshape(-1,), tied_tol=1e-08)[0]
            except Exception as e:
                print(e)
                c_index_val = 0.5

        if (epoch + 1) % 10 == 0:
            print('Epoch: {}, Train_loss: {:.4f}, Val_Loss: {:.4f}, Train_c_index: {:.4f}, Val_c_index: {:.4f}'.format(epoch + 1, total_loss, eval_final.item(), c_index, c_index_val))

        data = [epoch, total_loss, eval_final, c_index, c_index_val, cv]
        write_to_csv(os.path.join(csv_folder, filename), data, mode)

        mode = 'a'

        writer.add_scalar('train/loss', total_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)
        writer.add_scalar('val/loss', eval_final, epoch)
        writer.add_scalar('val/c_index', c_index_val, epoch)

    model_save_path = f'checkpoints_images/fold_{cv}_model.pth'
    torch.save(model.state_dict(), model_save_path)

    train_predictions = {
        'risk_scores': all_risk_scores,
        'censorships': all_censorships,
        'real_times': all_real_times
    }
    with open(f'checkpoints_images/fold_{cv}_train_predictions.pkl', 'wb') as f:
        pickle.dump(train_predictions, f)

    val_predictions = {
        'risk_scores': eval_risk_scores,
        'censorships': eval_censorships,
        'real_times': eval_real_times
    }
    with open(f'checkpoints_images/fold_{cv}_val_predictions.pkl', 'wb') as f:
        pickle.dump(val_predictions, f)

    print(f"Modello e predizioni del fold {cv} salvate!")

    writer.close()


def create_loader(args):
    if not args.cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda:1':
        torch.cuda.empty_cache()
        gc.collect()
    print('Device:', device)

    seed_torch(seed=42, device=device)

    mapping = generate_mapping(args)
    #modifed_data = create_bins(data=dataHum, label_col=None, n_bins=4, eps=1e-6)

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
            os.makedirs(os.path.join(args.logs, str(args.trials), f'cv_{cv}'), exist_ok=True)
        else:
            shutil.rmtree(os.path.join(args.logs, str(args.trials)))
            os.makedirs(os.path.join(args.logs, str(args.trials), f'cv_{cv}'), exist_ok=True)

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

            outcome_train = [1 if dataHum.loc[str(id), args.outcome] == 'Dead' or dataHum.loc[str(id), args.outcome] == 1 else 0 for id in ids_train if str(id) in mapping.keys()]
            outcome_val = [1 if dataHum.loc[str(id), args.outcome] == 'Dead' or dataHum.loc[str(id), args.outcome] == 1 else 0 for id in ids_val if str(id) in mapping.keys()]
            
            features_train = {str(id): stack_tensors(args, mapping[str(id)]) for id in ids_train if str(id) in mapping.keys() and str(id).lower() != "nan"}
            features_val = {str(id): stack_tensors(args, mapping[str(id)]) for id in ids_val if str(id) in mapping.keys() and str(id).lower() != "nan"}

            # times_train = [modifed_data.loc[str(id), 'time_label'] for id in ids_train if str(id) in mapping.keys()]
            # times_val = [modifed_data.loc[str(id), 'time_label'] for id in ids_val if str(id) in mapping.keys()]

            # real_times_train = [modifed_data.loc[str(id), 'OS censored at TPX  months'] for id in ids_train if str(id) in mapping.keys()]
            # real_times_val = [modifed_data.loc[str(id), 'OS censored at TPX  months'] for id in ids_val if str(id) in mapping.keys()]

            real_times_train = [dataHum.loc[str(id), 'OS censored at TPX  months'] for id in ids_train if str(id) in mapping.keys()]
            real_times_val = [dataHum.loc[str(id), 'OS censored at TPX  months'] for id in ids_val if str(id) in mapping.keys()]

            features_train_filtered, masking_train, to_remove_train = create_masking(features_train)
            features_val_filtered, masking_val, to_remove_val = create_masking(features_val)
            outcome_train_filtered = [value for index, value in enumerate(outcome_train) if index not in to_remove_train]
            outcome_val_filtered = [value for index, value in enumerate(outcome_val) if index not in to_remove_val]
            real_times_train_filtered = [value for index, value in enumerate(real_times_train) if index not in to_remove_train]
            real_times_val_filtered = [value for index, value in enumerate(real_times_val) if index not in to_remove_val]
            #times_train_filtered = [value for index, value in enumerate(times_train) if index not in to_remove_train]
            #times_val_filtered = [value for index, value in enumerate(times_val) if index not in to_remove_val]


            train_dataset = SurvivalDataset(list(features_train_filtered.keys()), list(features_train_filtered.values()), np.array(times_train_filtered), np.array(real_times_train_filtered), np.array(outcome_train_filtered), list(masking_train.values()))
            val_dataset = SurvivalDataset(list(features_val_filtered.keys()), list(features_val_filtered.values()), np.array(times_val_filtered), np.array(real_times_val_filtered), np.array(outcome_val_filtered), list(masking_val.values()))

            train_loader = DataLoader(train_dataset, batch_size=21, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=21, shuffle=False, drop_last=True)

            train_test_moma(args, train_loader, val_loader, writer, cv, device)
    
    plot_box(args)


if __name__ == '__main__':
    args = parser.parse_args()
    create_loader(args)