import os
import gc
import shutil
import numpy as np
import pandas as pd
import argparse
import pickle

from sksurv.metrics import concordance_index_censored

from tensorboardX import SummaryWriter

import sys
import warnings
sys.path.append('models')
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from seeds import seed_torch
from surv_integration import create_bins, write_to_csv, write_information_multiomics, plot_box_multiomics

from torch.utils.data import DataLoader
from models import MultiOmicsDF
from losses_dataset import NLLSurvLoss, CensoredCrossEntropyLoss, CoxLoss, WeibulLoss, MultiomicsDataset, WeibulLossContinuous, cox_ph_loss, LogCoshLoss

os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "2"


dataHum = pd.read_csv('dataset/MultiomicsFinal.csv', index_col='ID')
multiomics = pd.read_csv('dataset_multiomics/Input_MM_ICH_top2000_norm_Scaled.csv', index_col='ID')


parser = argparse.ArgumentParser(description='Survival Analysis similar to MOMA')
parser.add_argument('--outcome', type=str, default='Outcome at last FU', help='The column name for the outcome')
parser.add_argument('--logs', type=str, default='logs', help='The folder for the Tensorboard results')
parser.add_argument('--trials', type=int, default=1000, required=False, help='The trials that we are trying')
parser.add_argument('--epochs', type=int, default=30, help='The number of epochs for training')
parser.add_argument('--cuda', type=bool, default=True, help='Set to True if you want to use GPU')
parser.add_argument('--features', type=str, default='features', help='Folder for the features')


def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def train_test_moma(args, train_loader, val_loader, writer, cv, device):
    
    mode = 'w'
    filename = f"results_{cv}.csv"

    csv_folder = os.path.join('results_multiomics', str(args.trials))

    model = MultiOmicsDF(input_size=1536, dropout=0.4)
    model.to(device)

    criterions = {
        'CE' : nn.CrossEntropyLoss(reduction='mean'),
        'NLL' : NLLSurvLoss(alpha=0.8),
        'Censored' : CensoredCrossEntropyLoss(),
        'Cox' : CoxLoss(),
        'WB' : WeibulLoss(),
        'WBC': WeibulLossContinuous(),
        'VAE': LogCoshLoss(),
        'VAECox': cox_ph_loss
    }

    optimizers = {
        'AD': optim.Adam(model.parameters(), lr=1e-04, weight_decay=1e-03),
        'AW': optim.AdamW(model.parameters(), lr=1e-05, weight_decay=1e-05),
        'ASGD': optim.ASGD(model.parameters(), lr=1e-05, weight_decay=1e-04),
        'RMS': optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-05, momentum=0.5),
        'SGD': optim.SGD(model.parameters(), lr=1e-05, weight_decay=1e-04, momentum=0.9, nesterov=True)
    }

    criterion_cox = criterions['VAECox']
    optimizer = optimizers['RMS']

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min = 0, last_epoch = -1)

    seed_torch(seed=42, device=device)
    write_information_multiomics(args, os.path.join(csv_folder, 'trial_information.txt'), model, optimizer, args.epochs, scheduler)

    model.train()
    for epoch in range(args.epochs):
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

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=10, norm_type=2.0)
            torch.cuda.empty_cache()

        all_risk_scores = np.concatenate(all_risk_scores)
        all_censorships = np.concatenate(all_censorships)
        all_real_times = np.concatenate(all_real_times)

        try:
            c_index = concordance_index_censored((all_censorships.reshape(-1,)).astype(bool), all_real_times, all_risk_scores.reshape(-1,), tied_tol=1e-08)[0]
        except Exception as e:
            print(e)
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
                features_val, demog_val, clin_val, genomic_val, transcr_val, real_times_val, event_indicator_val, masking_val = features_val.to(device), demog_val.to(device), clin_val.to(device), genomic_val.to(device), transcr_val.to(device), real_times_val.to(device), event_indicator_val.to(device), masking_val.to(device)

                eval_risk = model(features_val, masking_val, demog_val, clin_val, genomic_val, transcr_val)
                loss_cox_test = criterion_cox(eval_risk, real_times_val, event_indicator_val) 
                eval_loss += loss_cox_test

                eval_risk_scores.append(eval_risk.detach().cpu().numpy())
                eval_censorships.append(event_indicator_val.detach().cpu().numpy())
                eval_real_times.append(real_times_val.detach().cpu().numpy())

                torch.cuda.empty_cache()

            eval_final = eval_loss /len(val_loader)

            eval_risk_scores = np.concatenate(eval_risk_scores)
            eval_censorships = np.concatenate(eval_censorships)
            eval_real_times = np.concatenate(eval_real_times)

            try:
                c_index_val = concordance_index_censored((eval_censorships.reshape(-1,)).astype(bool), eval_real_times, eval_risk_scores.reshape(-1,), tied_tol=1e-08)[0]
            except Exception as e:
                print(e)
                c_index_val = 0.5

        if (epoch + 1) % 10 == 0:
            print('Epoch: {}, Train_loss: {:.4f}, Val_Loss: {:.4f}, Train_c_index: {:.4f}, Val_c_index: {:.4f}'.format(epoch + 1, total_loss, eval_final.item(), c_index, c_index_val))

        data = [epoch, total_loss, eval_final.item(), c_index, c_index_val, cv]
        write_to_csv(os.path.join(csv_folder, filename), data, mode)

        mode = 'a'

        writer.add_scalar('train/loss', total_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)
        writer.add_scalar('val/loss', eval_final, epoch)
        writer.add_scalar('val/c_index', c_index_val, epoch)

    torch.save(model.state_dict(), os.path.join('checkpoints', f'model_{cv}.pth'))

    writer.close()



def create_loader(args):
    if not args.cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    
    if device.type == 'cuda:0':
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


    if not os.path.exists(os.path.join('results_multiomics', args.logs, str(args.trials))):
        os.makedirs(os.path.join('results_multiomics', args.logs, str(args.trials)), exist_ok=True)
    else:
        shutil.rmtree(os.path.join('results_multiomics', args.logs, str(args.trials)))
        os.makedirs(os.path.join('results_multiomics', args.logs, str(args.trials)), exist_ok=True)

    if not os.path.exists(os.path.join('results_multiomics', str(args.trials))):
        os.makedirs(os.path.join('results_multiomics', str(args.trials)), exist_ok=True)
    else:
        shutil.rmtree(os.path.join('results_multiomics', str(args.trials)))
        os.makedirs(os.path.join('results_multiomics', str(args.trials)), exist_ok=True)

    for cv, split in enumerate(os.listdir('splits')):
        if not os.path.exists(os.path.join('results_multiomics', args.logs, str(args.trials), f'cv_{cv}')):
            os.makedirs(os.path.join('results_multiomics', args.logs, str(args.trials), f'cv_{cv}'), exist_ok=True)
        else:
            shutil.rmtree(os.path.join('results_multiomics', args.logs, str(args.trials)))
            os.makedirs(os.path.join('results_multiomics', args.logs, str(args.trials), f'cv_{cv}'), exist_ok=True)

        writer = SummaryWriter(os.path.join('results_multiomics', args.logs, str(args.trials), f'cv_{cv}'), flush_secs=15)

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

            train_loader = DataLoader(train_dataset, batch_size=25, collate_fn=custom_collate_fn, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=25, collate_fn=custom_collate_fn, shuffle=False)

            train_test_moma(args, train_loader, val_loader, writer, cv, device)
    
    plot_box_multiomics(args)


if __name__ == '__main__':
    args = parser.parse_args()
    create_loader(args)