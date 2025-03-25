import os
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tensorly as tl
import torch.optim as optim
import matplotlib.pyplot as plt

from sksurv.metrics import concordance_index_censored

import sys
import warnings
sys.path.append('models')
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA

from focal_loss.focal_loss import FocalLoss
from losses_dataset import NLLSurvLoss, CensoredCrossEntropyLoss, CoxLoss
from models import CoxPH, Model2, ModelAtt, EasyModel


tl.set_backend('pytorch')
torch.set_num_threads(100)


dataHum = pd.read_csv('dataset/MultiomicsFinal.csv', index_col='ID')


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
            if args.complete:
                continue
            stacked_tensor = torch.cat((stacked_tensor, tensor.cpu()), dim=0)
    return stacked_tensor.detach()


def write_to_csv(filename, data, mode='a'):
    with open(filename, mode=mode, newline='') as file:
        writer = csv.writer(file)
        if mode == 'w':
            writer.writerow(['Epoch', 'TrainLoss', 'ValLoss', 'Train_C-Index', 'Val_C-Index', 'CV'])
        writer.writerow(data)


def write_information(args, path, model, optimizer, criterion, epochs, scheduler=None):
    with open(path, "w") as file:
        file.write("Model Information\n\n")
        file.write(f"Epochs --> {epochs}\n")
        file.write(f"Features --> {args.features}\n")
        file.write(f"Model Architecture --> {model}\n")
        file.write(f"Optimizer --> {optimizer}\n")
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


def plot_box(args):
    c_indexes = []

    for file in os.listdir(os.path.join('results_final', str(args.trials))):
        if file.endswith('.csv'):
            c_indexes.append(('_'.join(['Fold', file.split('.')[0].split('_')[-1]]), pd.read_csv(os.path.join('results_final', str(args.trials), file))['Val_C-Index']))

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
    plt.savefig(f'results_final/{str(args.trials)}/boxplot.png')


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


def train_test(args, train_loader, val_loader, writer, cv, device):

    mode = 'w'
    filename = f"results_{cv}.csv"

    csv_folder = os.path.join('results_final', str(args.trials))

    model = Model2(input_size=1536, n_classes=4, dropout=0.7)
    model.to(device)

    criterions = {
        'CE' : nn.CrossEntropyLoss(reduction='mean'),
        'NLL' : NLLSurvLoss(alpha=0),
        'Focal' : FocalLoss(gamma=4),
        'Censored' : CensoredCrossEntropyLoss(),
        'Cox' : CoxLoss()
    }

    optimizers = {
        'AD': optim.Adam(model.parameters(), lr=1e-04),
        'AW': optim.AdamW(model.parameters(), lr=1e-04, weight_decay=1e-05),
        'SGD': optim.SGD(model.parameters(), lr=9.347529720384705e-05, weight_decay=0.008034032759452312, momentum=0.9, nesterov=True)
    }

    criterion = criterions['Censored']
    optimizer = optimizers['SGD']

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, threshold=2e-05)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 0, last_epoch = -1)

    write_information(args, os.path.join(csv_folder, 'trial_information.txt'), model, optimizer, criterion, args.epochs)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0

        all_risk_scores = []
        all_censorships = []
        all_event_times = []

        for name, features, time_indicator, event_indicator in train_loader:
            if args.pca:
                features = features.squeeze(0).numpy()
                pca = PCA(n_components=300)
                features = torch.tensor(pca.fit_transform(features)).unsqueeze(0)

            features, time_indicator, event_indicator = features.to(device), time_indicator.to(device), event_indicator.to(device)

            optimizer.zero_grad()
            h = model(features)

            m = torch.nn.Softmax(dim=-1)
            loss = criterion(m(h), time_indicator)

            #loss = criterion(h, time_indicator, event_indicator)
            #loss = criterion(h, time_indicator.reshape(-1,))

            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()

            all_risk_scores.append(risk)
            all_censorships.append(event_indicator.detach().cpu().numpy())
            all_event_times.append(time_indicator.detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=10, norm_type=2.0)
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
                if args.pca:
                    features_val = features_val.numpy()
                    pca = PCA(n_components=300)
                    features_val = torch.tensor(pca.fit_transform(features_val))
                
                features_val, time_indicator_val, event_indicator_val = features_val.to(device), time_indicator_val.to(device), event_indicator_val.to(device)

                optimizer.zero_grad()
                h_val = model(features_val)

                m_eval = torch.nn.Softmax(dim=-1)
                eval_loss = criterion(m_eval(h_val), time_indicator_val)

                #eval_loss += criterion(h_val, time_indicator_val, event_indicator_val)
                #eval_loss += criterion(h_val, time_indicator_val.reshape(-1,))

                eval_hazards = torch.sigmoid(h_val)
                eval_survival = torch.cumprod(1 - eval_hazards, dim=1)
                eval_risk = -torch.sum(eval_survival, dim=1).detach().cpu().numpy()

                eval_risk_scores.append(eval_risk)
                eval_censorships.append(event_indicator_val.detach().cpu().numpy())
                eval_event_times.append(time_indicator_val.detach().cpu().numpy())

                torch.cuda.empty_cache()

            eval_final = eval_loss /len(val_loader)

            eval_risk_scores = np.concatenate(eval_risk_scores)
            eval_censorships = np.concatenate(eval_censorships)
            eval_event_times = np.concatenate(eval_event_times)

            c_index_val = concordance_index_censored((1-eval_censorships).astype(bool), eval_event_times, eval_risk_scores, tied_tol=1e-08)[0]

        if (epoch + 1) % 10 == 0:
            print('Epoch: {}, Train_loss: {:.4f}, Val_Loss: {:.4f}, Train_c_index: {:.4f}, Val_c_index: {:.4f}'.format(epoch + 1, total_loss, eval_final.item(), c_index, c_index_val))

        data = [epoch, total_loss, eval_final.item(), c_index, c_index_val, cv]
        write_to_csv(os.path.join(csv_folder, filename), data, mode)

        mode = 'a'

        writer.add_scalar('train/loss', total_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)
        writer.add_scalar('val/loss', eval_final, epoch)
        writer.add_scalar('val/c_index', c_index_val, epoch)
    
    torch.save(model.state_dict(), os.path.join(csv_folder, f'{cv}_checkpoint.pt'))

    writer.close()
