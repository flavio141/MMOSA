import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch import Tensor

import numpy as np


class SurvivalDataset(Dataset):
    def __init__(self, name, features, times, real_times, event_indicators, masking):
        self.name = name
        self.features = features
        self.times = times
        self.real_times = real_times
        self.event_indicators = event_indicators
        self.masking = masking
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.name[idx], self.features[idx], self.times[idx], self.real_times[idx], self.event_indicators[idx], self.masking[idx]


class SurvivalDatasetModified(Dataset):
    def __init__(self, features, labels, times, args):
        self.features = features
        self.labels = labels
        self.times = times
        self.patient_ids = list(features.keys())

        self.args = args
      
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_data = self.features[patient_id]
        stainings = None

        for image in patient_data:
            if stainings is None:
                stainings = torch.load(os.path.join(self.args.features, image))
            else:
                stainings = torch.cat((stainings, torch.load(os.path.join(self.args.features, image))), dim=0)
        
        return stainings, self.times[patient_id], self.labels[patient_id]


class MultiomicsDataset(Dataset):
    def __init__(self, data_dict, rows=15000):
        self.data_dict = data_dict
        self.patient_ids = list(data_dict.keys())
        self.to_remove = []
        self.rows = rows

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_data = self.data_dict[patient_id]

        demog_features = torch.tensor(np.array(patient_data['Demog'], dtype=float), dtype=torch.float32)
        clin_features = torch.tensor(np.array(patient_data['Clin'], dtype=float), dtype=torch.float32)
        genomic_features = torch.tensor(np.array(patient_data['Genomic'], dtype=float), dtype=torch.float32)
        transcr_features = torch.tensor(np.array(patient_data['Transcr'], dtype=float), dtype=torch.float32)

        stainings = None
        for key in patient_data['Stainings'].keys():
            if len(patient_data['Stainings'][key]) != 0:
                for image in patient_data['Stainings'][key]:
                    if stainings is None:
                        stainings = torch.load(os.path.join('features_giga', image))
                    else:
                        stainings = torch.cat((stainings, torch.load(os.path.join('features_giga', image))), dim=0)

        if patient_data['Outcomes'][0] == 'Dead' or patient_data['Outcomes'][0] == 'Death':
            outcome = torch.tensor([1.0], dtype=torch.float32)
        else:
            outcome = torch.tensor([0.0], dtype=torch.float32)

        time = torch.tensor(np.array(patient_data['time'], dtype=float), dtype=torch.float32)

        if stainings is not None and stainings.shape[0] > self.rows:
            self.to_remove.append(idx)
            return None

        features_padding = torch.full((self.rows, 1536), -999, dtype=stainings.dtype)
        if stainings is not None:
            features_padding[:stainings.shape[0], :] = stainings

        masking = torch.tensor(stainings.shape[0])

        sample = {
            'patient_id': patient_id,
            'demog': demog_features,
            'clin': clin_features,
            'genomic': genomic_features,
            'transcr': transcr_features,
            'staining': features_padding,
            'masking': masking,
            'outcome': outcome,
            'time': time
        }

        return sample


class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020)
    """

    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, c):
        y = y.type(torch.int64).reshape(-1, 1)
        c = c.type(torch.int64).reshape(-1, 1)
        hazards = torch.sigmoid(h)

        S = torch.cumprod(1 - hazards, dim=1)
        S_padded = torch.cat([torch.ones_like(c), S], 1)


        s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=self.eps)
        h_this = torch.gather(hazards, dim=1, index=y).clamp(min=self.eps)
        s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=self.eps)

        uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
        censored_loss = - c * torch.log(s_this)

        neg_l = censored_loss + uncensored_loss
        if self.alpha is not None:
            loss = (1 - self.alpha) * neg_l + self.alpha * uncensored_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            raise ValueError("Bad input for reduction: {}".format(self.reduction))
        return loss
    

EPS = 1e-12


class CensoredCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CensoredCrossEntropyLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y, obs):
        x = self.softmax(x).clamp(min=EPS)
        n = x.shape[0]
        loss_sum = 0
        for i in range(n):
            if obs[i] > 0.5:
                loss_sum += torch.log(x[i, y[i]])
            else:
                if y[i].item() == x.shape[1] - 1:
                    loss_sum += torch.log(x[i, y[i]].clamp(min=EPS))
                else:
                    loss_sum += torch.log(torch.sum(x[i, y[i]+1:]).clamp(min=EPS))
        loss_val = loss_sum / n * -1
        return loss_val
    


def partial_ll_loss(lrisks, survival_times, event_indicators):
    """
    lrisks: log risks, B x 1
    survival_times: time bin, B x 1
    event_indicators: event indicator, B x 1
    """
    
    num_uncensored = torch.sum(event_indicators, 0)
    if num_uncensored.item() == 0:
        return torch.sum(lrisks) * 0
    
    survival_times = survival_times.squeeze(1)
    event_indicators = event_indicators.squeeze(1)
    lrisks = lrisks.squeeze(1)

    sindex = torch.argsort(-survival_times)
    survival_times = survival_times[sindex]
    event_indicators = event_indicators[sindex]
    lrisks = lrisks[sindex]

    log_risk_stable = torch.logcumsumexp(lrisks, 0)

    likelihood = lrisks - log_risk_stable
    uncensored_likelihood = likelihood * event_indicators
    logL = -torch.sum(uncensored_likelihood)
    return logL / num_uncensored


class CoxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, logits, times, censorships):
        return partial_ll_loss(lrisks = logits, survival_times=times, event_indicators=(1-censorships).float())


def weibull_loglik_discrete(y_true, ab_pred):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]
    hazard0 = torch.pow((y_ + 1e-35) / a_, b_)
    hazard1 = torch.pow((y_ + 1) / a_, b_)
    return -1 * torch.mean(u_ * torch.log(torch.exp(hazard1 - hazard0) - 1.0) - hazard1)


class WeibulLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, ab_pred):
        return weibull_loglik_discrete(y_true, ab_pred)


def weib(x, a, n):
    return (a / n) * (x / n) ** (a - 1) * torch.exp(-(x / n) ** a)

def weibull_loglik_continuous(y_true, ab_pred):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]
    ya = (y_ + 1e-35) / a_
    return -1 * torch.mean(u_ * (torch.log(b_) + b_ * torch.log(ya)) - torch.pow(ya, b_))


class WeibulLossContinuous(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, ab_pred):
        return weibull_loglik_continuous(y_true, ab_pred)


def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())


def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7):
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)
    

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))