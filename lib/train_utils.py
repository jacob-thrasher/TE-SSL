import torchmetrics.functional as F
import torch
import numpy as np
import pandas as pd
from torch import nn
from lib.Loss import CoxLoss
from datasets.augmentations import BatchedAugment
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, integrated_brier_score
from lifelines.utils import concordance_index
from lib.utils import get_survival_curves_deephit
from pycox.evaluation import EvalSurv


def update_optim_deepsurv(optim, epoch, lr_decay):
    '''
    Updates optimizer as proposed in: 
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5828433/
    '''
    cur_lr = optim.param_groups[0]['lr']
    new_lr = cur_lr / (1 + (epoch*lr_decay))
    optim.param_groups[0]['lr'] = new_lr


# There is definitely a better way to wrap all three of these into one function, but for simplicity I will keep it as is
def train_step(model, dataloader, optim, loss_fn, lr_scheduler, warmup_scheduler=None, warmup_period=1000, device='cuda'):
    model.train()
    running_loss = 0

    augmenter = BatchedAugment()
    lrs = []
    for img, label, e, t, clin in tqdm(dataloader):
        img = augmenter(img, mode='train').to(device)
        label = label.to(device)
        e = e.to(device)
        t = t.to(device)
        clin = clin.to(device)

        optim.zero_grad()

        pred = model(img)

        loss = loss_fn(pred.squeeze(), t, e)
        running_loss += loss.item()

        loss.backward()
        optim.step()

        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    lr_scheduler.step()

        lrs.append(optim.param_groups[0]['lr'])       


    return running_loss / len(dataloader), lrs

def test_step(model, dataloader, loss_fn, device, evaluation='sksurv', return_surv=False):
    model.eval()
    running_loss = 0
    predictions = []
    events = []
    times = []

    augmenter = BatchedAugment()

    for img, label, e, t, clin in tqdm(dataloader):
        events += e.tolist()
        times += t.tolist()

        img = augmenter(img, mode='test').to(device)
        label = label.to(device)
        t = t.to(device)
        e = e.to(device)
        clin = clin.to(device)

        pred = model(img) # TODO: Create multimodal function with model(img, clin)

        predictions += pred.cpu().detach().tolist()
        # hazards += pred.cpu().detach().squeeze().tolist()
        loss = loss_fn(pred.squeeze(), t, e)
        running_loss += loss.item()

    # Create structured arrays for sksurv metric
    if evaluation == 'sksurv':
        labels = np.zeros(len(events), np.dtype({'names': ['cens', 'time'],
                                                'formats': ['?', '<f8']}))
        labels['cens'], labels['time'] = events, times
        C, _, _, _, _ = concordance_index_censored(events, times, predictions)
        ibs = -1 # Not computed

    elif evaluation == 'pycox':
        surv = get_survival_curves_deephit(torch.tensor(predictions))
        time_range = np.arange(0, 66, 6)
        ev = EvalSurv(pd.DataFrame(surv.T, time_range), np.array(times), np.array(events), censor_surv='km')

        C = ev.concordance_td('antolini')
        ibs = ev.integrated_brier_score(time_range)

        if return_surv: return running_loss / len(dataloader), C, ibs, surv, events, times
    
    else:
        raise NotImplementedError(f'Evaluation method {evaluation} not implemented! Try "sksurv" or "pycox"')
    
    return running_loss / len(dataloader), C, ibs

def train_step_with_reg(model, dataloader, optim, loss_fn, lr_scheduler, accum_iter=1, device='cuda', warmup_scheduler=None, warmup_period=1000):
    model.train()
    running_loss = 0

    augmenter = BatchedAugment()
    lrs = []
    for batch_idx, (img, _, e, t, age) in enumerate(tqdm(dataloader)):
        X_1 = augmenter(img, mode='train').to(device)
        X_2 = augmenter(img, mode='train').to(device)
        e = e.to(device)
        t = t.to(device)
        age = age.to(device)


        h_1, proj_1 = model(X_1)
        h_2, proj_2 = model(X_2)

        views = torch.cat([proj_1.unsqueeze(1), proj_2.unsqueeze(1)], dim=1)

        loss = loss_fn(pred=(h_1, h_2), t=t, e=e, features=views)
        running_loss += loss.item()

        loss = loss / accum_iter # Normalize loss to account for batch accumulation
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % accum_iter == 0 or (batch_idx + 1) == len(dataloader):
            optim.step()
            optim.zero_grad()

            if warmup_scheduler is not None:
                with warmup_scheduler.dampening():
                    if warmup_scheduler.last_step + 1 >= warmup_period:
                        lr_scheduler.step()

            lrs.append(optim.param_groups[0]['lr'])

    return running_loss / len(dataloader), lrs

def test_step_with_reg(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0
    hazards = []
    events = []
    times = []

    augmenter = BatchedAugment()

    for img, _, e, t, age in tqdm(dataloader):
        events += e.tolist()
        times += t.tolist()

        X_1 = augmenter(img, mode='test').to(device) # Don't augment first image for test step
        X_2 = augmenter(img, mode='train').to(device)
        e = e.to(device)
        t = t.to(device)
        age = age.to(device)

        h_1, proj_1 = model(X_1)
        h_2, proj_2 = model(X_2)

        hazards += h_1.cpu().detach().squeeze().tolist()

        views = torch.cat([proj_1.unsqueeze(1), proj_2.unsqueeze(1)], dim=1)
        loss = loss_fn(pred=(h_1, h_2), t=t, e=e, features=views)
        running_loss += loss.item()

    C, _, _, _, _ = concordance_index_censored(events, times, hazards)
    # C = 0
    return running_loss / len(dataloader), C

def train_step_contrastive(model, dataloader, optim, loss_fn, lr_scheduler, accum_iter=1, device='cuda', warmup_scheduler=None, warmup_period=1000, method='self'):
    
    assert method in ['self', 'supervised', 'TESSL'], f'Expected "method" to be one of [self, supervised, TESSL], got {method}'

    model.train()
    running_loss = 0

    augmenter = BatchedAugment()
    lrs = []
    for batch_idx, (img, _, e, t, _) in enumerate(tqdm(dataloader)):
        X_1 = augmenter(img, mode='train').to(device)
        X_2 = augmenter(img, mode='train').to(device)
        e = e.to(device)
        t = t.to(device)


        proj_1 = model(X_1)
        proj_2 = model(X_2)

        if method == 'self': 
            e = None
            t = None
        elif method == 'supervised': 
            t = None

        views = torch.cat([proj_1.unsqueeze(1), proj_2.unsqueeze(1)], dim=1)
        loss = loss_fn(features=views, labels=e, times=t, mask=None)
        running_loss += loss.item()

        loss = loss / accum_iter # Normalize loss to account for batch accumulation
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % accum_iter == 0 or (batch_idx + 1) == len(dataloader):
            optim.step()
            optim.zero_grad()

            if warmup_scheduler is not None:
                with warmup_scheduler.dampening():
                    if warmup_scheduler.last_step + 1 >= warmup_period:
                        lr_scheduler.step()

            lrs.append(optim.param_groups[0]['lr'])

    return running_loss / len(dataloader), lrs


def test_step_contrastive(model, dataloader, loss_fn, device, method='self'):

    assert method in ['self', 'supervised', 'TESSL'], f'Expected "method" to be one of [self, supervised, TESSL], got {method}'

    model.eval()
    running_loss = 0

    augmenter = BatchedAugment()

    for img, _, e, t, age in tqdm(dataloader):
        X_1 = augmenter(img, mode='train').to(device)
        X_2 = augmenter(img, mode='train').to(device)
        e = e.to(device)
        t = t.to(device)
        age = age.to(device)

        proj_1 = model(X_1)
        proj_2 = model(X_2)

        if method == 'self': 
            e = None
            t = None
        elif method == 'supervised': 
            t = None
            
        views = torch.cat([proj_1.unsqueeze(1), proj_2.unsqueeze(1)], dim=1)
        loss = loss_fn(features=views, labels=e, times=t, mask=None)
        running_loss += loss.item()

    return running_loss / len(dataloader)
