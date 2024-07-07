import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from torch.utils.data import DataLoader


class FullModel(nn.Module):
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss =  loss
    def forward(self, inputs, targets):
        outputs = self.model(inputs[0],inputs[1])
        loss = self.loss(outputs, targets)
        return torch.unsqueeze(loss,0),outputs

def pad_col(input, val=0, where='end'):
    """
    From: https://github.com/havakv/pycox/blob/master/pycox/models/utils.py
    Addes a column of `val` at the start of end of `input`.
    """
    if len(input.shape) != 2:
        raise ValueError(f"Only works for `phi` tensor that is 2-D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == 'end':
        return torch.cat([input, pad], dim=1)
    elif where == 'start':
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

def get_survival_curves_deephit(pred):
    '''
    From: https://github.com/havakv/pycox/blob/master/pycox/models/pmf.py
    Get survival curve for batch of predictions
    '''
    # Add extra column so cumsum != 1 at max(t) in final curve
    padded_pred = pad_col(pred)

    # Convert to probabilities
    pmf = nn.functional.softmax(padded_pred, dim=1)

    # Drop padded column
    pmf = pmf[:, :-1]

    # Cumsum and inverse probs
    surv = 1 - torch.cumsum(pmf, dim=1)
    return surv

##############
# DATA UTILS #
##############


def reduce_csvs(csvroot, dst, mriroot):
    df = pd.read_csv(csvroot, sep='\t')
    init_len = len(df)
    for i, row in df.iterrows():
        if not os.path.exists(os.path.join(mriroot, row.participant_id, row.session_id)):
            df.drop(index=i, inplace=True, axis=0)

    print(init_len, len(df))
    df.to_csv(dst, sep='\t')
    

def generate_new_tsv(tsvroot, subjectroot, dst):
    '''
    Generate new tsv for clinica. 

    Basically just input a the train/test/val tsv and
    only select the subjects that appear in subjectroot
    '''

    df = pd.read_csv(tsvroot, sep='\t')
    subjects = os.listdir(subjectroot)
    subjects = [x for x in subjects if 'sub-' in x]

    reduced = df.loc[df['participant_id'].isin(subjects)]
    reduced.to_csv(dst, sep='\t')

    print(f'Original: {len(df)}\nReduced: {len(reduced)}\n')

def fix_session_label(src, dst=None):
    '''
    The session_id column in the dataset is formatted differently than the
    session ids in the actual dataset (M006 vs M06). 
    This function addresses that problem
    '''
    if dst is None: dst = src

    df = pd.read_csv(src, sep='\t')

    for i, row in df.iterrows():
        sess = row.session_id
        df.loc[i, 'session_id'] = sess[:-2] + '0' + sess[-2:]
    
    df.drop('Unnamed: 0', axis=1)
    df.to_csv(dst, sep='\t', index=False)

def create_month_column(src, dst=None):
    '''
    Creates a month column M to store the month of the visit.
    This information is already available in the session_id, but
    adding this column makes accessing easier down the line
    '''
    
    if dst is None: dst = src

    df = pd.read_csv(src, sep='\t')
    df['M'] = -1 # Insert new column

    for i, row in df.iterrows():
        sess = row.session_id
        month = int(sess.split('M')[-1]) # ex. SES-M006 -> [SES-, 006] --> "006" --> 6
        df.loc[i, 'M'] = month

    df.to_csv(dst, sep='\t', index=False)

def create_exp_folder(root):
    '''
    Automaticaly creates a new experiment folder so nothing accidentally gets overwritten
    '''

    _id = 1
    exp_path = os.path.join(root, str(_id))
    while os.path.exists(exp_path):
        _id += 1
        exp_path = os.path.join(root, str(_id))

    os.mkdir(exp_path)
    return exp_path
