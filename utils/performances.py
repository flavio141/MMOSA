import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb

from models import AttVAE
from sksurv.metrics import concordance_index_censored

params = {
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


def fold_performance(train_loader, val_loader, fold, device):
    model = AttVAE().to(device)
    model.load_state_dict(torch.load(f"vae_model_{fold}.pth"))
    model.eval()

    mut_train = []
    logvar_train = []
    all_time = []
    all_event = []
    all_demog = []
    all_clin = []
    all_genomic = []
    all_transcr = []


    for train_sample in train_loader:
        features, demog, clin, genomic, transcr, real_time, event_indicator, masking = train_sample['staining'], train_sample['demog'], train_sample['clin'], train_sample['genomic'], train_sample['transcr'], train_sample['time'], train_sample['outcome'], train_sample['masking']
        features, masking = features, masking

        _, mu, logvar, _ = model(features, masking)
        mut_train.append(mu.detach().cpu().numpy())
        logvar_train.append(logvar.detach().cpu().numpy())
        all_time.append(real_time.numpy())
        all_event.append(event_indicator.numpy())
        all_demog.append(demog.numpy())
        all_clin.append(clin.numpy())
        all_genomic.append(genomic.numpy())
        all_transcr.append(transcr.numpy())
        
        features = features.detach()
        del features

        masking = masking.detach()
        del masking

        nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=10, norm_type=2.0)
        torch.cuda.empty_cache()


    mu_val_list = []
    logvar_val_list = []
    all_time_val = []
    all_event_val = []
    all_demog_val = []
    all_clin_val = []
    all_genomic_val = []
    all_transcr_val = []


    for val_sample in val_loader:
        features_val, demog_val, clin_val, genomic_val, transcr_val, real_times_val, event_indicator_val, masking_val = val_sample['staining'], val_sample['demog'], val_sample['clin'], val_sample['genomic'], val_sample['transcr'], val_sample['time'], val_sample['outcome'], val_sample['masking']
        features_val, masking_val = features_val, masking_val

        _, mu_val, logvar_val, _ = model(features_val, masking_val)
        all_time_val.append(real_times_val.numpy())
        all_event_val.append(event_indicator_val.numpy())
        all_demog_val.append(demog_val.numpy())
        all_clin_val.append(clin_val.numpy())
        all_genomic_val.append(genomic_val.numpy())
        all_transcr_val.append(transcr_val.numpy())
        

        mu_val_list.append(mu_val.detach().numpy())
        logvar_val_list.append(logvar_val.detach().numpy())

        torch.cuda.empty_cache()

    mu_train = np.concatenate(mut_train)
    logvar_train = np.concatenate(logvar_train)

    std_train = torch.exp(0.5 * torch.from_numpy(logvar_train))
    eps_train = torch.randn_like(std_train)
    z_train = mu_train + eps_train.numpy() * std_train.numpy()

    mu_val = np.concatenate(mu_val_list)
    logvar_val = np.concatenate(logvar_val_list)

    std_val = torch.exp(0.5 * torch.from_numpy(logvar_val))
    eps_val = torch.randn_like(std_val)
    z_val = mu_val + eps_val.numpy() * std_val.numpy()

    all_time = np.concatenate(all_time)
    all_event = np.concatenate(all_event)

    all_demog = np.concatenate(all_demog)
    all_clin = np.concatenate(all_clin)
    all_genomic = np.concatenate(all_genomic)
    all_transcr = np.concatenate(all_transcr)

    all_time_val = np.concatenate(all_time_val)
    all_event_val = np.concatenate(all_event_val)

    z_val = np.concatenate(mu_val_list)
    all_demog_val = np.concatenate(all_demog_val)
    all_clin_val = np.concatenate(all_clin_val)
    all_genomic_val = np.concatenate(all_genomic_val)
    all_transcr_val = np.concatenate(all_transcr_val)

    X_train = np.concatenate([z_train, all_demog, all_clin, all_genomic, all_transcr], axis=1)
    X_test = np.concatenate([z_val, all_demog_val, all_clin_val, all_genomic_val, all_transcr_val], axis=1)

    dtrain = xgb.DMatrix(X_train, label=all_time, base_margin=all_event)
    xgb_model = xgb.train(params, dtrain, num_boost_round=100)
    all_risk_cox = xgb_model.predict(dtrain)
    c_index = concordance_index_censored((all_event.reshape(-1,)).astype(bool), all_time, all_risk_cox.reshape(-1,), tied_tol=1e-08)[0]

    dtest = xgb.DMatrix(X_test)
    eval_risk_cox = xgb_model.predict(dtest)
    c_index_val = concordance_index_censored((all_event_val.reshape(-1,)).astype(bool), all_time_val, eval_risk_cox.reshape(-1,), tied_tol=1e-08)[0]

    print('Train_c_index: {:.4f}, Val_c_index: {:.4f}'.format(c_index, c_index_val))