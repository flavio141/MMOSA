import os
import gc
import shutil
import argparse
import numpy as np
import pandas as pd


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

from seeds import seed_torch
from surv_utils import generate_mapping, plot_box, write_to_csv, write_information

from torch.utils.data import DataLoader, default_collate
from models import ModelSurv, ModelSurvCustom
#from focal_loss.focal_loss import FocalLoss
from losses_dataset import SurvivalDatasetModified, cox_ph_loss

os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "2"


dataHum = pd.read_csv('dataset/MultiomicsFinal.csv', index_col='ID')
num_features = 1024

parser = argparse.ArgumentParser(description='Survival Analysis similar to MOMA')
parser.add_argument('--outcome', type=str, default='Outcome at last FU', help='The column name for the outcome')
parser.add_argument('--time', type=str, default='OS censored at TPX  months', help='The column name for the time')

parser.add_argument('--logs', type=str, default='logs', help='The folder for the Tensorboard results')
parser.add_argument('--trials', type=int, default=100, required=False, help='The trials that we are trying')
parser.add_argument('--epochs', type=int, default=50, help='The number of epochs for training')
parser.add_argument('--features', type=str, default='features_dino', help='Folder for the features')


def collate_fn(batch):
    batch = [f for f in batch if f[0].shape[0] <= 20000]

    if len(batch) == 0:
        return None, None, None, None
    
    max_rows = max(f[0].shape[0] for f in batch)
    #batch = [(torch.cat([variables[0], torch.full((max_rows - variables[0].shape[0], num_features), 0, dtype=variables[0].dtype, device=variables[0].device)]), torch.tensor(variables[0].shape[0], device=variables[0].device), variables[1], variables[2]) for variables in batch]    
    batch = [(torch.cat([variables[0], torch.full((max_rows - variables[0].shape[0], num_features), 0, dtype=variables[0].dtype, device=variables[0].device)]), variables[1], variables[2]) for variables in batch]        
    return default_collate(batch)


def train_test_custom(args, train_loader, val_loader, writer, cv, device):
    mode = 'w'
    filename = f"results_{cv}.csv"
    csv_folder = os.path.join('results_final', str(args.trials))

    model = ModelSurvCustom(input_size=num_features, surv_nodes=[num_features, 64, 1], dropout=0.4)
    model.to(device)

    criterions = {
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
        all_real_times = []


        for features, real_time, event_indicator in train_loader:
            if features is None:
                continue
            features, real_time, event_indicator = features.to(device), real_time.to(device), event_indicator.to(device)

            optimizer.zero_grad()
            risk = model(features)
            loss = criterion_cox(risk, real_time, event_indicator) 

            all_risk_scores.append(risk.detach().cpu().numpy())
            all_censorships.append(event_indicator.detach().cpu().numpy())
            all_real_times.append(real_time.detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            del features, real_time, event_indicator, risk, loss

            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2.0)
            torch.cuda.empty_cache()

        all_risk_scores = np.concatenate(all_risk_scores)
        all_censorships = np.concatenate(all_censorships)
        all_real_times = np.concatenate(all_real_times)

        c_index = concordance_index_censored((all_censorships).astype(bool), all_real_times, all_risk_scores.reshape(-1,), tied_tol=1e-08)[0]
        scheduler.step(total_loss)

        model.eval()
        with torch.no_grad():
            eval_loss = 0
            eval_risk_scores = []
            eval_censorships = []
            eval_real_times = []

            for features, real_time, event_indicator in val_loader:
                if features is None:
                    continue
                features, real_time, event_indicator = features.to(device), real_time.to(device), event_indicator.to(device)

                eval_risk = model(features)
                loss_cox_test = criterion_cox(eval_risk, real_time, event_indicator) 
                eval_loss += loss_cox_test

                eval_risk_scores.append(eval_risk.detach().cpu().numpy())
                eval_censorships.append(event_indicator.detach().cpu().numpy())
                eval_real_times.append(real_time.detach().cpu().numpy())

                del features, real_time, event_indicator, eval_risk, loss_cox_test

                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2.0)
                torch.cuda.empty_cache()

            eval_final = eval_loss /len(val_loader)

            eval_risk_scores = np.concatenate(eval_risk_scores)
            eval_censorships = np.concatenate(eval_censorships)
            eval_real_times = np.concatenate(eval_real_times)

            c_index_val = concordance_index_censored((eval_censorships).astype(bool), eval_real_times, eval_risk_scores.reshape(-1,), tied_tol=1e-08)[0]

        if (epoch + 1) % 5 == 0:
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

    print(f"Model and prediction saved for fold {cv}!")

    writer.close()


def create_loader(args):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    gc.collect()
    print('Device:', device)

    seed_torch(seed=42, device=device)

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

    if not os.path.exists(os.path.join('checkpoints_images')):
        os.makedirs(os.path.join('checkpoints_images'), exist_ok=True)

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
            ids_train = list(split_data['train'].dropna())
            ids_val = list(split_data['val'].dropna())

            features_train = generate_mapping(args, ids_train)
            features_val = generate_mapping(args, ids_val)           

            outcome_train = {str(id): dataHum.loc[str(id), args.outcome] for id in list(features_train.keys())}
            outcome_val = {str(id): dataHum.loc[str(id), args.outcome] for id in list(features_val.keys())}

            real_times_train = {str(id): dataHum.loc[str(id), args.time] for id in list(features_train.keys())}
            real_times_val = {str(id): dataHum.loc[str(id), args.time] for id in list(features_val.keys())}

            train_dataset = SurvivalDatasetModified(features_train, outcome_train, real_times_train, args)
            val_dataset = SurvivalDatasetModified(features_val, outcome_val, real_times_val, args)

            train_loader = DataLoader(train_dataset, batch_size=15, collate_fn=collate_fn, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=15, collate_fn=collate_fn, shuffle=False, drop_last=True)

            train_test_custom(args, train_loader, val_loader, writer, cv, device)
    
    plot_box(args)


if __name__ == '__main__':
    args = parser.parse_args()
    dataHum["Outcome at last FU"] = dataHum["Outcome at last FU"].map(lambda x: 1 if x in ['Death', 'Dead'] else 0)
    create_loader(args)