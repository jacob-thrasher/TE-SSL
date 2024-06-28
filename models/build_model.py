import numpy as np
import torch

from collections import OrderedDict
from .modules import OutputHead, DualHeadedProjection, FullModel
from .models import MRIEncoder, ClinicalDataEncoder, MultiModalFusion

def prepare_model(mri_out=128, head_hid=200, dropout=0.5, 
    head_out=3, expansion=8, norm_type='Instance', 
    activation='relu', head_type='Dual'):

    assert head_type in ['Dual', 'Single'], f'Expected param head_type to be in [Dual, Single], got {head_type}'

    # Create image embedding model
    image_embeding_model = MRIEncoder(in_channel=1, feat_dim=mri_out, expansion=expansion, norm_type=norm_type, activation=activation, dropout=dropout)

    # Create output head
    if head_type == 'Single':
        head = OutputHead(in_dim=mri_out, n_hid=head_hid, out_dim=head_out)
    else:
        head = DualHeadedProjection(in_dim=mri_out, n_hid=head_hid, proj_dim=head_out) # Outputs (1, head_out), where 1 is hazard coef, head_out is proj

    # Assemble final model
    main_model = FullModel(image_embeding_model, head)

    return main_model

def load_weights(model, weights_path):
    state_dict = torch.load(weights_path)
    new_dict = OrderedDict()

    # tbh I'm not sure why 'module.' appears in some keys, 
    # but this just gets rid of those
    for key in state_dict:
        value = state_dict[key]
        if 'module' in key:
            key = key.replace('module.', '')

        new_dict[key] = value

    model.load_state_dict(new_dict)

    return model