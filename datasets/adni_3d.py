import os, torch, pdb
import numpy as np
import json
from PIL import Image
from PIL import ImageFile
import torch.utils.data as data
import random 
import collections
from numpy import random as nprandom
import pickle
import glob
import re
import numpy as np
import pandas as pd
from random import shuffle
import random
import math
import nibabel as nib
from .augmentations import *
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchio as tio

class ADNI_3D(data.Dataset):

    # TODO: FIX TSV LOADING FUNCTION
    def __init__(self, 
                 dir_to_scans, 
                 dir_to_tsv, 
                 dir_type='bids', 
                 mode='Train', 
                 n_label=3, 
                 percentage_usage=1.0, 
                 label_type='future'):
        '''
        Args:
            dir_to_sans - path to folder containing MRI scans in BIDS format
            dir_to_tsv - path to tsv file (not sure what info this contains yet)
            dir_type - directory structure of MRI scans.
                        Value must be in ['bids', 'caps']
                        See clinica documentation for more information: 
                        https://aramislab.paris.inria.fr/clinica/docs/public/latest/
            mode - Train/test mode
            n_label - number of labels to include
                        n_label=2 --> CN and AD only
                        n_label=3 --> CN, MCI, AD
            percentage_usage - Percentage of data to use. This is for defining the size of the training set compared to test
        '''
        assert dir_type in ['bids', 'caps'], f'param "dir_type" must be in [bids, caps], got {dir_type}'

        # NOTE: "Stubby" and "Past" loaders not yet implemented
        assert label_type in ['future'], f'param "label_type" must be in [future], got {label_type}'

        # Define label mapping -- Don't need this, but won't delete yet
        if n_label == 3:
            LABEL_MAPPING = ["CN", "MCI", "AD"]
        elif n_label == 2:
            LABEL_MAPPING = ["CN", "AD"]
        self.LABEL_MAPPING = LABEL_MAPPING  



        subject_tsv = pd.io.parsers.read_csv(os.path.join(dir_to_tsv, f'{mode}_reduced.tsv'), sep='\t')

        # Clean sessions without labels
        indices_not_missing = []
        for i in range(len(subject_tsv)):
            if mode == 'Train':
                if (subject_tsv.iloc[i].diagnosis in LABEL_MAPPING):
                    indices_not_missing.append(i)
            else:
                if (subject_tsv.iloc[i].diagnosis in LABEL_MAPPING):
                    indices_not_missing.append(i)

        self.subject_tsv = subject_tsv.iloc[indices_not_missing]
        if mode == 'Train':
            self.subject_tsv = subject_tsv.iloc[np.random.permutation(int(len(subject_tsv)*percentage_usage))]

        # Get all subject IDs from dataset
        self.subject_id = np.unique(subject_tsv.participant_id.values)

        # Corresponding indices for each subject I think?
        self.index_dic = dict(zip(self.subject_id,range(len(self.subject_id))))
        self.dir_to_scans = dir_to_scans

        self.label_type = label_type
        self.dir_type = dir_type
        self.mode = mode
        self.n_label = n_label
        self.age_range = list(np.arange(0.0, 120.0, 0.5)) # This part has to do with age embedding in models.py

        template_target = tio.ScalarImage(os.path.join(dir_to_tsv, 'TPM.nii'))
        self.mni_transform = tio.transforms.Resample(template_target)

    def centerCrop(self, img, length, width, height):
        assert img.shape[1] >= length
        assert img.shape[2] >= width
        assert img.shape[3] >= height

        x = img.shape[1]//2 - length//2
        y = img.shape[2]//2 - width//2
        z = img.shape[3]//2 - height//2
        img = img[:,x:x+length, y:y+width, z:z+height]
        return img

    def randomCrop(self, img, length, width, height):
        assert img.shape[1] >= length
        assert img.shape[2] >= width
        assert img.shape[3] >= height

        x = random.randint(0, img.shape[1] - length)
        y = random.randint(0, img.shape[2] - width)
        z = random.randint(0, img.shape[3] - height )
        img = img[:,x:x+length, y:y+width, z:z+height]
        return img
    
    def augment_image(self, image):
        sigma = np.random.uniform(0.0,1.0,1)[0]
        image = scipy.ndimage.filters.gaussian_filter(image, sigma, truncate=8)
        return image

    def unpickling(self, path):
       file_return=pickle.load(open(path,'rb'))
       return file_return

    def __len__(self):
        return len(self.subject_tsv)

    def __getitem__(self, idx):
        '''
        Returns:
            X     - Features
            label - AD classifiction label (0: CN/MCI, 1: AD)
            e     - Event indicator
            t     - time of event/censor
            age   - encoded age vector
        '''
        
        # ------------------------
        # CLASSIFIER CODE
        dx = self.subject_tsv.iloc[idx].diagnosis
        if self.n_label == 2:
            if dx == 'CN' or dx == 'MCI':
                label = 0
            elif dx == 'AD':
                label = 1
            else:
                label = -100 # Missing labels
        elif self.n_label == 3:
            if dx == 'CN': label = 0
            elif dx == 'MCI': label = 1
            elif dx == 'AD': label = 2
            else: label = -100

        # # Get Mini Mental State Exam score --> used to help validate model outputs
        # # There should be some correlation btwn MMSE score and AD
        # mmse = self.subject_tsv.iloc[idx].mmse
        # cdr_sub = 0#self.subject_tsv.iloc[idx].cdr #cdr_sb #cdr#

        # # Get index of age 
        # age = list(np.arange(0.0,120.0,0.5)).index(self.subject_tsv.iloc[idx].age_rounded) #list(np.arange(0.0,25.0)).index(self.subject_tsv.iloc[idx].education_level)#
        # idx_out = self.index_dic[self.subject_tsv.iloc[idx].participant_id]
            
        # --------------------------------
        # CLINICAL DATA
        row = self.subject_tsv.iloc[idx]
        # TODO: make this a parameter
        data_columns = ['AGE', 'PTEDUCAT', 'ADAS11', 'ADAS13', 'CDRSB', 'FAQ',
                        'LDELTOTAL', 'MMSE', 'RAVLT_forgetting', 'RAVLT_immediate',
                        'RAVLT_learning', 'RAVLT_perc_forgetting']
        clinical_data = torch.tensor(row[data_columns].tolist()).type(torch.float32)
        

        # --------------------------------
        # SURVIVAL ANALYSIS DATA

        # Get event indicator
        if self.label_type == 'future':
            # Get subject's history and sort by M
            participant_id = self.subject_tsv.iloc[idx].participant_id
            history = self.subject_tsv[self.subject_tsv['participant_id'] == participant_id].reset_index()
            history.sort_values(by=['M'], inplace=True)

            # Get date of first AD DX, or censor
            events = history['diagnosis'].to_list()
            if 'AD' in events:
                e = True # TODO: For Pycox, change to float
                event_time = history.iloc[events.index("AD")].M
            else:
                e = False # TODO: For Pycox, change to float
                event_time = history.iloc[-1].M # Date of last visit

            t = max(0, event_time - self.subject_tsv.iloc[idx].M) # t = months until positive DX or censor


        # --------------------------------
        # LOAD MRI
                        
        # Build path:
        if self.dir_type == 'caps':
            # root -> subject -> session -> segmentation
            path = os.path.join(self.dir_to_scans, self.subject_tsv.iloc[idx].participant_id,
                    self.subject_tsv.iloc[idx].session_id,'t1/spm/segmentation/normalized_space') 
        else:
            # root -> subject -> session -> scan
            path = os.path.join(self.dir_to_scans, self.subject_tsv.iloc[idx].participant_id,
                    self.subject_tsv.iloc[idx].session_id,'anat') 
        all_segs = list(os.listdir(path))

        try:
            for seg_name in all_segs:
                if 'Space_T1w' in seg_name or self.dir_type == 'bids':
                    # image = nib.load(os.path.join(path,seg_name)).get_fdata().squeeze()
                    image = tio.ScalarImage(os.path.join(path, seg_name)).numpy()
        

        except Exception as ex:
            print(f"Failed to load #{idx}: {path}")
            raise ex
        
        return torch.tensor(image.astype(np.float32)), label, e, t, clinical_data
        # return torch.tensor(image.astype(np.float32)), (torch.tensor(t), torch.tensor(e))