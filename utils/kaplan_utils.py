import os
import csv
import torch
import survhive
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

from models import Model3
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from focal_loss.focal_loss import FocalLoss
from losses_dataset import NLLSurvLoss, CensoredCrossEntropyLoss, CoxLoss, WeibulLoss


tl.set_backend('pytorch')
torch.set_num_threads(100)


dataHum = pd.read_csv('dataset_mgg/MultiomicsFinal.csv', index_col='ID')


def write_to_csv(filename, data, mode='a'):
    with open(filename, mode=mode, newline='') as file:
        writer = csv.writer(file)
        if mode == 'w':
            writer.writerow(['Epoch', 'TrainLoss', 'ValLoss', 'Train_C-Index', 'Val_C-Index', 'CV'])
        writer.writerow(data)


def write_information(args, path, model, optimizer, criterion, epochs):
    with open(path, "w") as file:
        file.write("Model Information\n\n")
        file.write(f"Epochs --> {epochs}\n")
        file.write(f"Features --> {args.features}\n")
        file.write(f"Model Architecture --> {model}\n")
        file.write(f"Optimizer --> {optimizer}\n")
        file.write(f"Loss Function --> {criterion}\n")
        file.write("\n")
        file.write("Dataset Information\n\n")
        file.write(f"Dataset --> {args.features}\n")
        file.write(f"Outcome --> {args.outcome}\n")
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


def train_test(args, train_loader, val_loader, writer, cv, device):

    mode = 'w'
    filename = f"results_{cv}.csv"
    csv_folder = os.path.join('results_final', str(args.trials))

    model = Model3(input_size=1536, dropout=0.6)
    model.to(device)

    criterions = {
        'CE' : nn.CrossEntropyLoss(reduction='mean'),
        'NLL' : NLLSurvLoss(alpha=0.8),
        'Focal' : FocalLoss(gamma=4),
        'Censored' : CensoredCrossEntropyLoss(),
        'Cox' : CoxLoss(),
        'WB' : WeibulLoss()
    }

    optimizers = {
        'AD': optim.Adam(model.parameters(), lr=1e-04, weight_decay=1e-05),
        'AW': optim.AdamW(model.parameters(), lr=1e-04, weight_decay=1e-05),
        'ASGD': optim.ASGD(model.parameters(), lr=1e-04, weight_decay=1e-05),
        'SGD': optim.SGD(model.parameters(), lr=9.347529720384705e-05, weight_decay=0.008034032759452312, momentum=0.9, nesterov=True)
    }

    criterion = criterions['Focal']
    optimizer = optimizers['AD']

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, threshold=2e-05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 0, last_epoch = -1)

    write_information(args, os.path.join(csv_folder, 'trial_information.txt'), model, optimizer, criterion, args.epochs)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0

        all_risk_scores = []
        all_censorships = []
        all_event_times = []
        all_data = []
        all_real_times = []

        #cox = CoxPHSurvivalAnalysis()
        trace = survhive.CoxNet(rng_seed=42)

        for _, features, time_indicator, real_time, event_indicator in train_loader:
            if features.size(1) > 10000:
                continue
            features, time_indicator, real_time, event_indicator = features.to(device), time_indicator.to(device), real_time.to(device), event_indicator.to(device)

            optimizer.zero_grad()
            h = model(features)

            m = torch.nn.Softmax(dim=-1)
            loss = criterion(m(h), time_indicator)

            #loss = criterion(h, time_indicator, event_indicator)

            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()


            all_risk_scores.append(risk)
            all_data.append(h.detach().cpu().numpy())
            all_censorships.append(event_indicator.detach().cpu().numpy())
            all_event_times.append(time_indicator.detach().cpu().numpy())
            all_real_times.append(real_time.detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=10, norm_type=2.0)
            torch.cuda.empty_cache()

        all_data = np.concatenate(all_data)
        all_risk_scores = np.concatenate(all_risk_scores)
        all_censorships = np.concatenate(all_censorships)
        all_event_times = np.concatenate(all_event_times)
        all_real_times = np.concatenate(all_real_times)

        X_train = pd.DataFrame(all_data)
        y_train = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_real_times)

        trace.fit(X_train, y_train)
        #cox.fit(X_train, y_train)

        c_index = concordance_index_censored(y_train['event'], y_train['time'], trace.predict(X_train), tied_tol=1e-08)[0]
        scheduler.step(total_loss)

        model.eval()
        with torch.no_grad():
            eval_loss = 0
            eval_risk_scores = []
            eval_censorships = []
            eval_event_times = []
            eval_data = []
            eval_real_times = []

            for _, features_val, time_indicator_val, real_times_val, event_indicator_val in val_loader:
                if features_val.size(1) > 10000:
                    continue
                features_val, time_indicator_val, real_times_val, event_indicator_val = features_val.to(device), time_indicator_val.to(device), real_times_val.to(device), event_indicator_val.to(device)

                optimizer.zero_grad()
                h_val = model(features_val)

                m_eval = torch.nn.Softmax(dim=-1)
                eval_loss += criterion(m_eval(h_val), time_indicator_val)
                #eval_loss += criterion(h_val, time_indicator_val, event_indicator_val)

                eval_hazards = torch.sigmoid(h_val)
                eval_survival = torch.cumprod(1 - eval_hazards, dim=1)
                eval_risk = -torch.sum(eval_survival, dim=1).detach().cpu().numpy()

                eval_risk_scores.append(eval_risk)
                eval_data.append(h_val.detach().cpu().numpy())
                eval_censorships.append(event_indicator_val.detach().cpu().numpy())
                eval_event_times.append(time_indicator_val.detach().cpu().numpy())
                eval_real_times.append(real_times_val.detach().cpu().numpy())

                torch.cuda.empty_cache()

            eval_final = eval_loss /len(val_loader)

            eval_data = np.concatenate(eval_data)
            eval_risk_scores = np.concatenate(eval_risk_scores)
            eval_censorships = np.concatenate(eval_censorships)
            eval_event_times = np.concatenate(eval_event_times)
            eval_real_times = np.concatenate(eval_real_times)

            X_test = pd.DataFrame(eval_data)
            y_test = Surv.from_arrays(event=(1 - eval_censorships).astype(bool), time=eval_real_times)

            c_index_val = concordance_index_censored(y_test['event'], y_test['time'], trace.predict(X_test), tied_tol=1e-08)[0]

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
