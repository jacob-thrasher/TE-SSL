# -*- coding: utf-8 -*-
"""
@author: Sheng
"""
import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init
import math
import sys
sys.path.append('Utils')

class MRIEncoder(nn.Module):

    def __init__(self, 
                 in_channel =1, 
                 feat_dim   = 1024,
                 expansion  = 4,
                 dropout    = 0.5,  
                 norm_type  = 'Instance', 
                 activation ='relu'):
        super(MRIEncoder, self).__init__()
    
        assert activation in ['relu', 'selu'], f'Expected param "activation" to be in [relu, selu], got {activation}'

        self.feat_dim = feat_dim            # For MultiModalFusion class

        if activation == 'relu': activation_fn = nn.ReLU(inplace=True)
        else: activation_fn = nn.SELU(inplace=True)

        self.conv = nn.Sequential()

        # BLOCK 1
        self.conv.add_module('conv0_s1',nn.Conv3d(in_channel, 4*expansion, kernel_size=1))

        if norm_type == 'Instance':
           self.conv.add_module('lrn0_s1',nn.InstanceNorm3d(4*expansion))
        else:
           self.conv.add_module('lrn0_s1',nn.BatchNorm3d(4*expansion))

        self.conv.add_module('relu0_s1', activation_fn)
        self.conv.add_module('pool0_s1',nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('dropout_1', nn.Dropout(dropout))


        # BLOCK 2
        self.conv.add_module('conv1_s1',nn.Conv3d(4*expansion, 32*expansion, kernel_size=3, padding=0, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn1_s1',nn.InstanceNorm3d(32*expansion))
        else:
            self.conv.add_module('lrn1_s1',nn.BatchNorm3d(32*expansion))
            
        self.conv.add_module('relu1_s1', activation_fn)
        self.conv.add_module('pool1_s1',nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('dropout_2', nn.Dropout(dropout))

        # BLOCK 3
        self.conv.add_module('conv2_s1',nn.Conv3d(32*expansion, 64*expansion, kernel_size=5, padding=2, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn2_s1',nn.InstanceNorm3d(64*expansion))
        else:
            self.conv.add_module('lrn2_s1',nn.BatchNorm3d(64*expansion))
        self.conv.add_module('relu2_s1', activation_fn)
        self.conv.add_module('pool2_s1',nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('dropout_3', nn.Dropout(dropout))

        # BLOCK 4
        self.conv.add_module('conv3_s1',nn.Conv3d(64*expansion, 64*expansion, kernel_size=3, padding=1, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn3_s1',nn.InstanceNorm3d(64*expansion))
        else:
            self.conv.add_module('lrn2_s1',nn.BatchNorm3d(64*expansion))
        self.conv.add_module('relu3_s1', activation_fn)
        self.conv.add_module('pool2_s1',nn.MaxPool3d(kernel_size=5, stride=2))

        self.conv.add_module('dropout_4', nn.Dropout(dropout))

        # PROJECTION
        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(64*expansion*5*5*5, feat_dim))


    def load(self,checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)['state_dict']
        pretrained_dict = {k[6:]: v for k, v in list(pretrained_dict.items()) if k[6:] in model_dict and 'conv3_s1' not in k and 'fc6' not in k and 'fc7' not in k and 'fc8' not in k}

        model_dict.update(pretrained_dict)
        

        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])
        return pretrained_dict.keys()

    def freeze(self, pretrained_dict_keys):
        for name, param in self.named_parameters():
            if name in pretrained_dict_keys:
                param.requires_grad = False
                

    def save(self,checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def forward(self, x):
        z = self.conv(x)
        z = self.fc6(z.view(x.shape[0],-1))

        return z

def weights_init(model):
    if type(model) in [nn.Conv3d, nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.1)