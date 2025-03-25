from tensorboardX import SummaryWriter

import os
import gc
import shutil
import numpy as np
import pandas as pd
import argparse
import pickle
import csv

import sys
import warnings
sys.path.append('models')
warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest

from torch.utils.data.dataloader import default_collate
from sksurv.metrics import concordance_index_censored
from lifelines.utils import concordance_index
from sksurv.util import Surv
from itertools import product

from seeds import seed_torch
from surv_utils import create_bins
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from torch.utils.data import DataLoader
from models_omics import ModelImages
from losses_dataset import MultiomicsDataset, cox_ph_loss

os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "2"


dataHum = pd.read_csv('dataset/MultiomicsFinal.csv', index_col='ID')
multiomics = pd.read_csv('dataset_multiomics/Input_MM_ICH_top2000_norm_Scaled.csv', index_col='ID')


params_xgboost = {
    'objective': 'survival:cox',
    'eval_metric': 'cox-nloglik',
    'eta': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1.0,
    'alpha': 0,
    'gamma': 0,
    'min_child_weight': 1,
    'scale_pos_weight': 2
}

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


parser = argparse.ArgumentParser(description='Survival Analysis similar to MOMA')
parser.add_argument('--outcome', type=str, default='Outcome at last FU', help='The column name for the outcome')
parser.add_argument('--logs', type=str, default='logs', help='The folder for the Tensorboard results')
parser.add_argument('--trials', type=int, default=1000, required=False, help='The trials that we are trying')
parser.add_argument('--epochs', type=int, default=20, help='The number of epochs for training')
parser.add_argument('--cuda', type=bool, default=True, help='Set to True if you want to use GPU')
parser.add_argument('--features', type=str, default='features', help='Folder for the features')
parser.add_argument('--xgboost', type=bool, default=False, help='Folder for the features')
parser.add_argument('--grid_search', type=bool, default=False, help='Perform Grid Search for XGBoost')


def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def write_information_multiomics(args, path, model_img, optimizer_img, criterion, epochs, scheduler=None):
    with open(path, "w") as file:
        file.write("Model Information\n\n")
        file.write(f"Epochs --> {epochs}\n")
        file.write(f"Model Images --> {model_img}\n")
        file.write(f"Optimizer Images --> {optimizer_img}\n")
        file.write(f"Loss Function --> {criterion}\n")
        file.write(f"Scheduler --> {scheduler}\n")
        file.write("\n")
        file.write("Dataset Information\n\n")
        file.write(f"Dataset --> {args.features}\n")
        file.write(f"Outcome --> {args.outcome}\n")
        file.write("\n")
        file.write("Argoments Information\n\n")
        file.write(f"Args --> {args}\n")
        file.write("\n")


def write_to_csv(filename, data, mode='a'):
    with open(filename, mode=mode, newline='') as file:
        writer = csv.writer(file)
        if mode == 'w':
            writer.writerow(['Epoch', 'TrainLossImage', 'ValLossImage', 'Train_C-Index', 'Val_C-Index', 'CV'])
        writer.writerow(data)


def plot_box_multiomics(args):
    c_indexes = []

    for file in os.listdir(os.path.join('results_multiomics', str(args.trials))):
        if file.endswith('.csv'):
            c_indexes.append(('_'.join(['Fold', file.split('.')[0].split('_')[-1]]), pd.read_csv(os.path.join('results_multiomics', str(args.trials), file))['Val_C-Index']))

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
    plt.savefig(f'results_multiomics/{str(args.trials)}/boxplot.png')


def concordance_index_scorer(event, time, predictions):
    c_index, _ = concordance_index_censored(event.astype(bool), time, predictions)
    return c_index


def xgboost_grid_search(X_train, X_test, all_time, all_event, all_time_val, all_event_val):
    param_grid_xgboost = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    c_index_ref = 0.5

    keys, values = zip(*param_grid_xgboost.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    for par in combinations:
        cv_c_indices = []

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X_train, all_event):
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = all_time[train_idx], all_time[val_idx]
            e_cv_train, e_cv_val = all_event[train_idx], all_event[val_idx]

            dtrain = xgb.DMatrix(X_cv_train, label=y_cv_train, base_margin=e_cv_train)
            xgb_model = xgb.train(par, dtrain, num_boost_round=100)
            all_risk_cox = xgb_model.predict(dtrain)
            c_index = concordance_index_censored((e_cv_train.reshape(-1,)).astype(bool), y_cv_train, all_risk_cox.reshape(-1,), tied_tol=1e-08)[0]

            dtest = xgb.DMatrix(X_cv_val)
            eval_risk_cox = xgb_model.predict(dtest)
            c_index_val = concordance_index_censored((e_cv_val.reshape(-1,)).astype(bool), y_cv_val, eval_risk_cox.reshape(-1,), tied_tol=1e-08)[0]
            cv_c_indices.append(1 - c_index_val)
        
        if np.mean(cv_c_indices) > c_index_ref:
            c_index_ref = np.mean(cv_c_indices)
            best_params = par
                
    dtrain = xgb.DMatrix(X_train, label=all_time, base_margin=all_event)
    xgb_model = xgb.train(best_params, dtrain, num_boost_round=100)
    all_risk_cox = xgb_model.predict(dtrain)
    c_index = concordance_index_censored((all_event.reshape(-1,)).astype(bool), all_time, all_risk_cox.reshape(-1,), tied_tol=1e-08)[0]

    dtest = xgb.DMatrix(X_test)
    eval_risk_cox = xgb_model.predict(dtest)
    c_index_val = concordance_index_censored((all_event_val.reshape(-1,)).astype(bool), all_time_val, eval_risk_cox.reshape(-1,), tied_tol=1e-08)[0]
    print('\nXGBoost --> Train_C_Index Total: {:.4f}, Val_C_Index Total: {:.4f}'.format(round(1 - c_index, 4), round(1 - c_index_val, 4)))


def xgboost_train_test(all_demog, all_clin, all_genomic, all_transcr, representation_complete, 
                       all_demog_val, all_clin_val, all_genomic_val, all_transcr_val, representation_val_complete,
                       all_time, all_event, all_time_val, all_event_val, cv):
    save = True
    best_c_index_cv = 0.5
    
    all_demog = np.concatenate(all_demog)
    all_clin = np.concatenate(all_clin)
    all_genomic = np.concatenate(all_genomic)
    all_transcr = np.concatenate(all_transcr)
    representation_complete = np.concatenate(representation_complete)

    all_demog_val = np.concatenate(all_demog_val)
    all_clin_val = np.concatenate(all_clin_val)
    all_genomic_val = np.concatenate(all_genomic_val)
    all_transcr_val = np.concatenate(all_transcr_val)
    representation_val_complete = np.concatenate(representation_val_complete)

    X_train = np.concatenate([all_demog, all_clin, all_genomic, all_transcr, representation_complete], axis=1)
    X_test = np.concatenate([all_demog_val, all_clin_val, all_genomic_val, all_transcr_val, representation_val_complete], axis=1)

    if save:
        np.save(f'matrix/X_train_{cv}.npy', X_train)
        np.save(f'matrix/X_test_{cv}.npy', X_test)
        np.save(f'matrix/all_time_{cv}.npy', all_time)
        np.save(f'matrix/all_event_{cv}.npy', all_event)
        np.save(f'matrix/all_time_val_{cv}.npy', all_time_val)
        np.save(f'matrix/all_event_val_{cv}.npy', all_event_val)

    if args.xgboost and not args.grid_search:
        dtrain = xgb.DMatrix(X_train, label=all_time, base_margin=all_event)
        xgb_model = xgb.train(params_xgboost, dtrain, num_boost_round=100)
        all_risk_cox = xgb_model.predict(dtrain)
        c_index = concordance_index_censored((all_event.reshape(-1,)).astype(bool), all_time, all_risk_cox.reshape(-1,), tied_tol=1e-08)[0]

        dtest = xgb.DMatrix(X_test)
        eval_risk_cox = xgb_model.predict(dtest)
        c_index_val = concordance_index_censored((all_event_val.reshape(-1,)).astype(bool), all_time_val, eval_risk_cox.reshape(-1,), tied_tol=1e-08)[0]
        print('\nXGBoost --> Train_C_Index Total: {:.4f}, Val_C_Index Total: {:.4f}'.format(round(c_index, 2), round(c_index_val, 2)))
    elif args.xgboost and args.grid_search:
        xgboost_grid_search(X_train, X_test, all_time, all_event, all_time_val, all_event_val)
    else:
        train_y = Surv.from_arrays(event=(all_event.reshape(-1,)).astype(bool), time=all_time)

        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in product(*values)]

        for combo in combinations:
            cv_c_indices = []

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(X_train, train_y):
                X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                y_cv_train, y_cv_val = train_y[train_idx], train_y[val_idx]

                rsf = RandomSurvivalForest(
                    n_estimators=combo["n_estimators"],
                    max_depth=combo["max_depth"],
                    min_samples_split=combo["min_samples_split"],
                    min_samples_leaf=combo["min_samples_leaf"],
                    random_state=42,
                )

                rsf.fit(X_cv_train, y_cv_train)
                risk_cv_val = rsf.predict(X_cv_val)

                c_index_cv = concordance_index_censored(event_indicator=y_cv_val["event"], event_time=y_cv_val["time"], estimate=risk_cv_val)[0]
                cv_c_indices.append(c_index_cv)

            mean_c_index_cv = np.mean(cv_c_indices)

            if mean_c_index_cv > best_c_index_cv:
                best_c_index_cv = mean_c_index_cv
                best_params = combo

        rsf_final = RandomSurvivalForest(**best_params)
        rsf_final.fit(X_train, train_y)

        all_risk_cox = rsf_final.predict(X_train)
        eval_risk_cox = rsf_final.predict(X_test)

        c_index = concordance_index_censored((all_event.reshape(-1,)).astype(bool), all_time, all_risk_cox)[0]
        c_index_val = concordance_index_censored((all_event_val.reshape(-1,)).astype(bool), all_time_val, eval_risk_cox)[0]
        
        print('\nRandomForest Final --> Train_C_Index Total: {:.4f}, Val_C_Index Total: {:.4f}, Params: {}'.format(round(c_index, 2), round(c_index_val, 2), best_params))



def train_test(args, train_loader, val_loader, writer, cv, device):
    
    mode = 'w'
    filename = f"results_{cv}.csv"

    csv_folder = os.path.join('results_multiomics', str(args.trials))

    model_img = ModelImages(input_size=1536, surv_nodes=[1536, 64, 1], dropout=0.3)
    model_img.to(device)

    criterion_images = cox_ph_loss
    optimizer_images = optim.Adam(model_img.parameters(), lr=1e-05, weight_decay=1e-04)

    scheduler_images = optim.lr_scheduler.CosineAnnealingLR(optimizer_images, T_max = 100, eta_min = 0, last_epoch = -1)

    seed_torch(seed=42, device=device)
    write_information_multiomics(args, os.path.join(csv_folder, 'trial_information.txt'), model_img, optimizer_images, args.epochs, scheduler_images)

    model_img.train()
    for epoch in range(args.epochs):
        total_loss_img = 0

        all_risk_img_scores = []
        all_censorships = []
        all_real_times = []

        if (epoch + 1) == args.epochs:
            demog_complete = []
            clin_complete = []
            genomic_complete = []
            transcr_complete = []
            representation_complete = []


        for train_sample in train_loader:
            features, demog, clin, genomic, transcr, real_time, event_indicator, masking = train_sample['staining'], train_sample['demog'], train_sample['clin'], train_sample['genomic'], train_sample['transcr'], train_sample['time'], train_sample['outcome'], train_sample['masking']
            features, real_time, event_indicator, masking = features.to(device), real_time.to(device), event_indicator.to(device), masking.to(device)
            
            optimizer_images.zero_grad()

            risk_img, representation = model_img(features, masking)
            loss_img = criterion_images(risk_img, real_time, event_indicator)

            all_risk_img_scores.append(risk_img.detach().cpu().numpy())
            all_censorships.append(event_indicator.detach().cpu().numpy())
            all_real_times.append(real_time.detach().cpu().numpy())

            loss_img.backward()
            optimizer_images.step()

            total_loss_img += loss_img.item()

            if (epoch + 1) == args.epochs:
                demog_complete.append(demog)
                clin_complete.append(clin)
                genomic_complete.append(genomic)
                transcr_complete.append(transcr)
                representation_complete.append(representation.detach().cpu().numpy())

            torch.cuda.empty_cache()

        all_risk_img_scores = np.concatenate(all_risk_img_scores)
        all_censorships = np.concatenate(all_censorships)
        all_real_times = np.concatenate(all_real_times)

        try:
            c_index = concordance_index_censored((all_censorships.reshape(-1,)).astype(bool), all_real_times, all_risk_img_scores.reshape(-1,), tied_tol=1e-08)[0]
        except Exception as e:
            print(e)
            c_index = 0.5

        scheduler_images.step(total_loss_img)

        model_img.eval()
        with torch.no_grad():
            eval_loss_img = 0

            eval_risk_img_scores = []
            eval_censorships = []
            eval_real_times = []

            if (epoch + 1) == args.epochs:
                demog_val_complete = []
                clin_val_complete = []
                genomic_val_complete = []
                transcr_val_complete = []
                representation_val_complete = []

            for val_sample in val_loader:
                features_val, demog_val, clin_val, genomic_val, transcr_val, real_times_val, event_indicator_val, masking_val = val_sample['staining'], val_sample['demog'], val_sample['clin'], val_sample['genomic'], val_sample['transcr'], val_sample['time'], val_sample['outcome'], val_sample['masking']
                features_val, real_times_val, event_indicator_val, masking_val = features_val.to(device), real_times_val.to(device), event_indicator_val.to(device), masking_val.to(device)

                eval_risk_img, representation_val = model_img(features_val, masking_val)
                loss_img_test = criterion_images(eval_risk_img, real_times_val, event_indicator_val) 


                eval_risk_img_scores.append(eval_risk_img.detach().cpu().numpy())
                eval_censorships.append(event_indicator_val.detach().cpu().numpy())
                eval_real_times.append(real_times_val.detach().cpu().numpy())

                eval_loss_img += loss_img_test

                if (epoch + 1) == args.epochs:
                    demog_val_complete.append(demog_val)
                    clin_val_complete.append(clin_val)
                    genomic_val_complete.append(genomic_val)
                    transcr_val_complete.append(transcr_val)
                    representation_val_complete.append(representation_val.detach().cpu().numpy())

                torch.cuda.empty_cache()

            eval_final_img = eval_loss_img /len(val_loader)

            eval_risk_img_scores = np.concatenate(eval_risk_img_scores)
            eval_censorships = np.concatenate(eval_censorships)
            eval_real_times = np.concatenate(eval_real_times)

            try:
                c_index_val = concordance_index_censored((eval_censorships.reshape(-1,)).astype(bool), eval_real_times, eval_risk_img_scores.reshape(-1,), tied_tol=1e-08)[0]
            except Exception as e:
                print(e)
                c_index_val = 0.5

        if (epoch + 1) % 5 == 0:
            print('Epoch: {}, Train_loss_img: {:.4f}, Val_Loss_img: {:.4f}'.format(epoch + 1, total_loss_img, eval_final_img.item()))
            print('NN Images --> Train_C-Index Images: {:.4f}, Val_C-Index Images: {:.4f}'.format(round(c_index, 4), round(c_index_val, 4)))

        data = [epoch, total_loss_img, eval_final_img.item(), c_index, c_index_val, cv]
        write_to_csv(os.path.join(csv_folder, filename), data, mode)

        mode = 'a'

        writer.add_scalar('train/loss_img', total_loss_img, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)
        writer.add_scalar('val/loss_img', eval_final_img, epoch)
        writer.add_scalar('val/c_index', c_index_val, epoch)


    # representation_complete = []
    # representation_val_complete = []

    # model_img = model_img.to(torch.device('cpu'))
    # model_img.eval()

    # with torch.no_grad():
    #     for train_sample in train_loader:
    #         features, masking = train_sample['staining'], train_sample['masking']

    #         _, representation = model_img(features, masking)

    #         representation_complete.append(representation.detach().cpu().numpy())

    #     for val_sample in val_loader:
    #         features_val, masking_val = val_sample['staining'], val_sample['masking']

    #         _, representation_val = model_img(features_val, masking_val)

    #         representation_val_complete.append(representation_val.detach().cpu().numpy())

    xgboost_train_test(demog_complete, clin_complete, genomic_complete, transcr_complete, representation_complete, 
                       demog_val_complete, clin_val_complete, genomic_val_complete, transcr_val_complete, representation_val_complete,
                       all_real_times, all_censorships, eval_real_times, eval_censorships, cv=cv)

    torch.save(model_img.state_dict(), f"matrix/model_img_{cv}.pth")

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

            train_loader = DataLoader(train_dataset, batch_size=21, collate_fn=custom_collate_fn, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=21, collate_fn=custom_collate_fn, shuffle=False, drop_last=True)

            train_test(args, train_loader, val_loader, writer, cv, device)
    
    plot_box_multiomics(args)


if __name__ == '__main__':
    args = parser.parse_args()
    create_loader(args)