import os
import gc
import csv
import torch
import optuna
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt

import sys
import warnings
sys.path.append('utils')
sys.path.append('models')
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored
from seeds import seed_torch
from losses_dataset import SurvivalDataset, NLLSurvLoss, CensoredCrossEntropyLoss
from models import CoxPH, Model2, ModelAtt, Mil_Attention

from focal_loss.focal_loss import FocalLoss

torch.set_num_threads(100)


dataHum = pd.read_csv('dataset/MultiomicsFinal.csv', index_col='ID')
parser = argparse.ArgumentParser(description='Survival Analysis')
parser.add_argument('--outcome', type=str, default='Outcome at last FU', help='The column name for the outcome')
parser.add_argument('--epochs', type=int, default=100, help='The number of epochs for training')
parser.add_argument('--cuda', type=bool, default=True, help='Set to True if you want to use GPU')
parser.add_argument('--features', type=str, default='features_vahadane', help='Folder for the features')


def generate_mapping(args):
    mapping = {}

    for id in dataHum.index:
        slides_features = os.listdir(args.features)
        
        if "_dp" in id:
            id = id[:-3]
        
        mapping[id] = []

        for slide in slides_features:
            if (os.path.join(args.features, slide).endswith('.pt')) and (id == slide.split('_')[0]):
                mapping[id].append(slide)

    return {k: v for k, v in mapping.items() if len(v) != 0}


def stack_tensors(args, tensor_files):
    stacked_tensor = None
    for file in tensor_files:
        tensor = torch.load(os.path.join(args.features, file))
        if stacked_tensor is None:
            stacked_tensor = tensor.cpu()
        else:
            stacked_tensor = torch.cat((stacked_tensor, tensor.cpu()), dim=0)
    return stacked_tensor.detach()


def write_to_csv(filename, data, mode='a'):
    with open(filename, mode=mode, newline='') as file:
        writer = csv.writer(file)
        if mode == 'w':
            writer.writerow(['Epoch', 'TrainLoss', 'ValLoss', 'Train_C-Index', 'Val_C-Index', 'CV'])
        writer.writerow(data)


def write_information(path, model, optimizer, criterion, epochs):
    with open(path, "w") as file:
        file.write("Model Information\n\n")
        file.write(f"Epochs --> {epochs}\n")
        file.write(f"Model Architecture --> {model}\n")
        file.write(f"Optimizer --> {optimizer}\n")
        file.write(f"Loss Function --> {criterion}\n")
        file.write("\n")


def plot_box(model_name, trial):
    c_indexes = []

    for file in os.listdir(os.path.join('optuna', model_name)):
        if f'trial{str(trial)}' in file and file.endswith('.csv'):
            c_indexes.append(('_'.join(['Fold', file.split('.')[0].split('_')[-1]]), pd.read_csv(os.path.join('optuna', model_name, file))['Val_C-Index']))

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
    plt.savefig(f'optuna/{model_name}/trial{str(trial)}_boxplot.png')


def create_bins(data=None, label_col=None, n_bins=5, eps=1e-6):
    if data is None:
        raise ValueError('Dataset in .csv format is required')

    if not label_col:
        label_col = 'OS censored at TPX  months'
    else:
        assert label_col in data.columns

    patients_df = data.copy()
    uncensored_df = patients_df[patients_df['Outcome at last FU'] == 'Dead']

    disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
    q_bins[-1] = data[label_col].max() + eps
    q_bins[0] = data[label_col].min() - eps

    disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
    patients_df.insert(2, 'time_label', disc_labels.values.astype(int))

    return patients_df


def train_test_optuna(args, train_loader, val_loader, parameters, trial, cv, device):

    mode = 'w'
    filename = f"trial{str(trial)}_results_{cv}.csv"

    csv_folder = os.path.join('optuna', parameters['model_name'])

    if parameters['model_name'] == 'CoxPH':
        model = CoxPH(input_size=1536, n_classes=4, dropout=parameters['dropout'])
    elif parameters['model_name'] == 'Model2':
        model = Model2(input_size=1536, n_classes=4, dropout=parameters['dropout'])
    elif parameters['model_name'] == 'ModelAtt':
        model = ModelAtt(input_size=1536, n_classes=4, dropout=parameters['dropout'])
    elif parameters['model_name'] == 'Mil_Attention':
        model = Mil_Attention(input_size=1536, n_classes=4, dropout=parameters['dropout'])
    else:
        raise ValueError(f"Unknown model name: {parameters['model_name']}")

    model.to(device)
    if parameters['loss_name'] == 'NLLSurvLoss':
        criterion = NLLSurvLoss(alpha=parameters['alpha'])
    elif parameters['loss_name'] == 'FocalLoss':
        criterion = FocalLoss(gamma=parameters['gamma'])
    else:
        criterion = CensoredCrossEntropyLoss()

    if parameters['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'])
    elif parameters['optimizer_name'] == 'RMS':
        optimizer = optim.RMSprop(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'])
    elif parameters['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'], momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer name: {parameters['optimizer_name']}")

    write_information(os.path.join(csv_folder, f'trial{str(trial)}_information.txt'), model, optimizer, criterion, parameters['epochs'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, threshold=2e-05)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = parameters['epochs'], eta_min = 0, last_epoch = -1)

    c_indexes = {}

    model.train()
    for epoch in range(parameters['epochs']):
        total_loss = 0

        all_risk_scores = []
        all_censorships = []
        all_event_times = []

        for name, features, time_indicator, event_indicator in train_loader:
            features, time_indicator, event_indicator = features.to(device), time_indicator.to(device), event_indicator.to(device)

            optimizer.zero_grad()
            h = model(features)
            m = torch.nn.Softmax(dim=-1)
            loss = criterion(m(h), time_indicator)

            #loss = criterion(h, time_indicator, event_indicator)

            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()

            all_risk_scores.append(risk)
            all_censorships.append(event_indicator.detach().cpu().numpy())
            all_event_times.append(time_indicator.detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            torch.cuda.empty_cache()

        all_risk_scores = np.concatenate(all_risk_scores)
        all_censorships = np.concatenate(all_censorships)
        all_event_times = np.concatenate(all_event_times)

        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        scheduler.step(total_loss)

        model.eval()
        with torch.no_grad():
            eval_loss = 0
            eval_risk_scores = []
            eval_censorships = []
            eval_event_times = []

            for name_val, features_val, time_indicator_val, event_indicator_val in val_loader:
                features_val, time_indicator_val, event_indicator_val = features_val.to(device), time_indicator_val.to(device), event_indicator_val.to(device)

                optimizer.zero_grad()
                h_val = model(features_val)
                #eval_loss += criterion(h_val, time_indicator_val, event_indicator_val)
                m_eval = torch.nn.Softmax(dim=-1)
                eval_loss = criterion(m_eval(h_val), time_indicator_val)

                hazards = torch.sigmoid(h_val)
                survival = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(survival, dim=1).detach().cpu().numpy()

                eval_risk_scores.append(risk)
                eval_censorships.append(event_indicator_val.detach().cpu().numpy())
                eval_event_times.append(time_indicator_val.detach().cpu().numpy())

                torch.cuda.empty_cache()

            eval_final = eval_loss /len(val_loader)

            eval_risk_scores = np.concatenate(eval_risk_scores)
            eval_censorships = np.concatenate(eval_censorships)
            eval_event_times = np.concatenate(eval_event_times)

            c_index_val = concordance_index_censored((1-eval_censorships).astype(bool), eval_event_times, eval_risk_scores, tied_tol=1e-08)[0]
            c_indexes[epoch] = c_index_val

        if (epoch + 1) % 10 == 0:
            print('Epoch: {}, Train_loss: {:.4f}, Val_Loss: {:.4f}, Train_c_index: {:.4f}, Val_c_index: {:.4f}'.format(epoch + 1, total_loss, eval_final.item(), c_index, c_index_val))
        
        data = [epoch, total_loss, eval_final.item(), c_index, c_index_val, cv]
        write_to_csv(os.path.join(csv_folder, filename), data, mode)

        mode = 'a'
    return np.mean(list(c_indexes.values())[-10:])#np.mean(list(c_indexes.values()))


def optuna_loader(args, parameters, trial):
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

    if not os.path.exists(os.path.join('optuna', parameters['model_name'])):
        os.makedirs(os.path.join('optuna', parameters['model_name']), exist_ok=True)

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

            outcome_train = [1 if dataHum.loc[str(id), args.outcome] == 'Dead' else 0 for id in ids_train if str(id) in mapping.keys()]
            outcome_val = [1 if dataHum.loc[str(id), args.outcome] == 'Dead' else 0 for id in ids_val if str(id) in mapping.keys()]

            features_train = {str(id): stack_tensors(args, mapping[str(id)]) for id in ids_train if str(id) in mapping.keys() and str(id).lower() != "nan"}
            features_val = {str(id): stack_tensors(args, mapping[str(id)]) for id in ids_val if str(id) in mapping.keys() and str(id).lower() != "nan"}

            times_train = [modifed_data.loc[str(id), 'time_label'] for id in ids_train if str(id) in mapping.keys()]
            times_val = [modifed_data.loc[str(id), 'time_label'] for id in ids_val if str(id) in mapping.keys()]
    
            train_dataset = SurvivalDataset(list(features_train.keys()), list(features_train.values()), np.array(times_train), np.array(outcome_train))
            val_dataset = SurvivalDataset(list(features_val.keys()), list(features_val.values()), np.array(times_val), np.array(outcome_val))

            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            c_indexes[cv] = train_test_optuna(args, train_loader, val_loader, parameters, trial, cv, device)
    
    plot_box(parameters['model_name'], trial)
    return c_indexes


def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    #alpha = trial.suggest_uniform('alpha', 0.5, 1.0)
    model_name = trial.suggest_categorical('model_name', ['ModelAtt'])
    optimizer_name = trial.suggest_categorical('optimizer_name', ['Adam', 'RMS', 'SGD'])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.7)
    loss_name = trial.suggest_categorical('loss_name', ['FocalLoss'])
    #threshold = trial.suggest_loguniform('threshold', 1e-5, 1e-4)
    #factor = trial.suggest_uniform('factor', 0.1, 0.9)
    #patience = trial.suggest_uniform('patience', 1, 10)
    epochs = trial.suggest_int('epochs', 10, 200)
    gamma = trial.suggest_int('gamma', 2, 5)

    args = parser.parse_args()
    parameters = {
        'lr': lr,
        'weight_decay': weight_decay,
        #'alpha': alpha,
        'model_name': model_name,
        'optimizer_name': optimizer_name,
        'dropout': dropout,
        'loss_name': loss_name,
        #'threshold': threshold,
        #'factor': factor,
        #'patience': patience,
        'gamma': gamma,
        'epochs': epochs
    }

    c_indexes = optuna_loader(args, parameters, trial._trial_id)
    return np.mean(list(c_indexes.values()))


if __name__ == '__main__':
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))