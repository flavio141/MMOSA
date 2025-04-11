import os
import gc
import shap
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import sys
sys.path.append('utils')
sys.path.append('models')

from seeds import seed_torch
from models import MultiOmicsDF
from surv_utils import create_bins
from losses_dataset import MultiomicsDataset


dataHum = pd.read_csv('dataset_new/MultiomicsFinal.csv', index_col='ID')
multiomics = pd.read_csv('dataset_multiomics/Input_MM_ICH_top2000_norm_Scaled.csv', index_col='ID')

parser = argparse.ArgumentParser(description='SHAP Explainability')
parser.add_argument('--extract_shap', type=bool, default=True, help='Extract SHAP values')


def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_model(model_path, device):
    model = MultiOmicsDF(input_size=1536, dropout=0.4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def compute_shap(model, data_loader, device):
    batch = next(iter(data_loader))

    features = batch['staining'].to(device).float()
    demog = batch['demog'].to(device).float()
    clin = batch['clin'].to(device).float()
    genomic = batch['genomic'].to(device).float()
    transcr = batch['transcr'].to(device).float()
    masking = batch['masking'].to(device).float()

    background_data = [features, masking, demog, clin, genomic, transcr]

    explainer = shap.DeepExplainer(model, background_data)
    shap_values = []

    for batch in tqdm(data_loader, desc='Computing SHAP values'):
        features = batch['staining'].to(device).float()
        demog = batch['demog'].to(device).float()
        clin = batch['clin'].to(device).float()
        genomic = batch['genomic'].to(device).float()
        transcr = batch['transcr'].to(device).float()
        masking = batch['masking'].to(device).float()

        val_data = [features, masking, demog, clin, genomic, transcr]
        shap_values_batch = explainer.shap_values(val_data, check_additivity=False)
        shap_values.append(shap_values_batch)
    
    grouped = zip(*shap_values)
    return [np.concatenate(shap_value, axis=0) for shap_value in grouped]


def prepare_dataset(args):
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


    for cv, split in enumerate(os.listdir('splits')):
        if split.endswith('.csv') and cv != 3 and cv != 4:
            if not os.path.exists(os.path.join('explainability', f'shap_fold_{cv}')):
                os.makedirs(os.path.join('explainability', f'shap_fold_{cv}'), exist_ok=True)
            
            print(f'\nCross Validation: Fold {cv}')
            split_data = pd.read_csv(f'splits/{split}')
            ids_val = split_data['val']

            if ids_val.dtype == 'float64':
                ids_val = ids_val.astype('Int64')

            val_dict = {patient_id: dataset_dict[patient_id] for patient_id in ids_val if patient_id in dataset_dict and '_dp' not in patient_id}
            val_dataset = MultiomicsDataset(val_dict, rows=15000)
            val_loader = DataLoader(val_dataset, batch_size=5, collate_fn=custom_collate_fn, shuffle=False)

            if args.extract_shap:
                model_path = f'final_model/model_{cv}.pth'
                model = load_model(model_path, device)
                model.to(device)
                print("Computing SHAP values...")
                shap_values = compute_shap(model, val_loader, device)
                
                for idx, value in enumerate(shap_values):
                    if idx != 1 and idx != 0:
                        np.save(f'explainability/shap_fold_{cv}/shap_value_{idx}.npy', value)
                shap_values = [value for idx, value in enumerate(shap_values) if idx != 1 and idx != 0]
            else:
                shap_values = [np.load(f'explainability/shap_fold_{cv}/shap_value_{idx}.npy') for idx in range(6) if idx != 1 and idx != 0]
            
            print("Visualizing SHAP values...")

            dataset = val_loader.dataset
            demog = torch.stack([sample['demog'] for sample in dataset if sample is not None]).numpy()
            clin = torch.stack([sample['clin'] for sample in dataset if sample is not None]).numpy()
            genomic = torch.stack([sample['genomic'] for sample in dataset if sample is not None]).numpy()
            transcr = torch.stack([sample['transcr'] for sample in dataset if sample is not None]).numpy()

            data = [demog, clin, genomic, transcr]
            feature_names = {
                0: [idx.replace('.', ' ').replace('_', ' ') for idx in list(multiomics.columns)[2:4]],
                1: [idx.replace('.', ' ').replace('_', ' ') for idx in list(multiomics.columns)[4:8]],
                2: [idx.replace('.', ' ').replace('_', ' ') for idx in list(multiomics.columns)[8:48]],
                3: [idx.replace('.', ' ').replace('_', ' ') for idx in list(multiomics.columns)[59:2081]]
            }

            for idx, value in tqdm(enumerate(shap_values), desc='Visualizing SHAP values'):
                if idx in feature_names: 
                    shap.summary_plot(value, data[idx], feature_names=feature_names[idx], show=False)
                    plt.savefig(f'explainability/shap_fold_{cv}/shap_plot_{idx}.png', format="png", dpi=300)
                    plt.close()

if __name__ == '__main__':
    args = parser.parse_args()
    prepare_dataset(args)
