import os
from torch.utils.data import DataLoader
from datasets.adni_3d import ADNI_3D
from models.build_model import *
from lib.train_utils import test_step, test_step_with_reg
from lib.Loss import CoxLoss, RegularizedCoxLoss, DeepHitSingleLoss

root = '/home/jacob/Documents/ADNI'
model_path = 'figures/deephit/10'
weights = os.path.join(model_path, 'best_model.pt')

test_dataset = ADNI_3D(dir_to_scans=os.path.join(root, 'mni'),
                    dir_to_tsv=os.path.join(root, 'tabular_data'),
                    dir_type='bids',
                    mode='Test',
                    n_label=3,
                    percentage_usage=1)

test_dataloader = DataLoader(test_dataset, batch_size=4)

model = prepare_model(mri_out=1024, head_hid=512, head_out=11, head_type='Single', multimodal=False)
model = load_weights(model, weights)

model.to('cuda')
model.eval()

loss, C, ibs, surv, events, times = test_step(model, test_dataloader, DeepHitSingleLoss(), 'cuda', evaluation='pycox', return_surv=True)
print(C)
print(ibs)

