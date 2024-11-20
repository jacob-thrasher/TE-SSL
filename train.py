import torch
import yaml
import os
import argparse
import matplotlib.pyplot as plt
import pytorch_warmup as warmup
import random
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, sampler
from torchlars import LARS
from datasets.adni_3d import ADNI_3D
from models.build_model import *
from lib.utils import create_exp_folder
from lib.Loss import RegularizedCoxLoss, SupConLoss, DeepHitSingleLoss, TESSL_Loss
from models.models import weights_init
from lib.train_utils import *
from models.modules import OutputHead

def train_model(cfg):
    print("Manual seed:", cfg['seed'])
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])



    exp_id = create_exp_folder(f'figures/{cfg["output_folder"]}')
    with open(os.path.join(exp_id, 'config.yaml'), 'w') as f:
        f.write(yaml.dump(cfg))


    # Create dataset
    print(f"Batch size: {cfg['data']['batch_size']}")
    print(f"--> Gradient Accumulation: {cfg['training_parameters']['accum_iter']} batches")
    print('--> Simulated batch size:', cfg['data']['batch_size'] * cfg['training_parameters']['accum_iter'])
    root = cfg['data']['data_root_dir']
    train_dataset = ADNI_3D(dir_to_scans=os.path.join(root, 'mni'),
                        dir_to_tsv=os.path.join(root, 'tabular_data'),
                        dir_type='bids',
                        mode='Train',
                        n_label=2,
                        percentage_usage=1)

    test_dataset = ADNI_3D(dir_to_scans=os.path.join(root, 'mni'),
                        dir_to_tsv=os.path.join(root, 'tabular_data'),
                        dir_type='bids',
                        mode='Val',
                        n_label=2,
                        percentage_usage=1)


    train_dataloader = DataLoader(train_dataset, 
                                batch_size=cfg['data']['batch_size'], 
                                shuffle=True, 
                                num_workers=cfg['data']['workers'], 
                                pin_memory=True)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=cfg['data']['batch_size'], 
                                shuffle=False, 
                                num_workers=cfg['data']['workers'],
                                drop_last=True)



    # Prepare model
    model = prepare_model(mri_out=cfg['model']['mri_out'], 
                        expansion=cfg['model']['expansion'], 
                        activation=cfg['model']['activation'],
                        head_out=cfg['model']['head_out'],
                        head_hid=cfg['model']['head_hid'], 
                        head_type=cfg['model']['head_type'],
                        dropout=cfg['model']['dropout'])

    if cfg['training_parameters']['pretrain'] is None:
        print("Intializing untrained model")
        weights_init(model)
    else:
        print('Loading pretrained model:', cfg['training_parameters']['pretrain'])
        model = load_weights(model, cfg['training_parameters']['pretrain'])

        model.head = OutputHead(in_dim=cfg['model']['mri_out'], n_hid=cfg['model']['head_hid'], out_dim=11)



    # To gpu
    device = cfg['primary_device'] if torch.cuda.is_available() else 'cpu'

    if device == 'cuda': devices = [0, 1, 2, 3]
    elif device == 'cuda:0': devices = [0, 1]
    else: devices = [2, 3]

    model = nn.DataParallel(model, device_ids=devices)
    model.to(device)
    print(f"Using cuda devices : {devices}")



    # Define loss
    paradigm = cfg['training_parameters']['paradigm']
    print("Training paradigm:", cfg['training_parameters']['paradigm'])
    if paradigm == 'Cox':
        loss_fn = CoxLoss()
    elif paradigm == 'CoxReg':
        loss_fn = RegularizedCoxLoss(alpha=cfg['training_parameters']['alpha'], 
                                    beta=cfg['training_parameters']['beta'], 
                                    do_supcon=cfg['training_parameters']['do_supcon'])
        print('Alpha =', cfg['training_parameters']['alpha'], ", Beta =", cfg['training_parameters']['beta'])
    elif paradigm == 'Contrastive':
        print(f"--> Method: {cfg['training_parameters']['contrastive_type']}")
        print(f"--> Alpha: {cfg['training_parameters']['alpha']}, Beta: {cfg['training_parameters']['beta']}")
        loss_fn = TESSL_Loss(temperature=0.07, base_temperature=0.07, alpha=cfg['training_parameters']['alpha'], beta=cfg['training_parameters']['beta'])
    elif paradigm == 'DeepHit':
        loss_fn = DeepHitSingleLoss(device=device)



    # Define optimizer
    print("Optimizer:", cfg['optimizer']['optim'])
    selected_optim = cfg['optimizer']['optim']
    if selected_optim == 'SGD':
        optim = SGD(model.parameters(), 
                        lr=cfg['optimizer']['lr'], 
                        momentum=cfg['optimizer']['momentum'])
    elif selected_optim == 'LARS':
        optim = SGD(model.parameters(), 
                        lr=cfg['optimizer']['lr'], 
                        momentum=cfg['optimizer']['momentum'])
        optim = LARS(optim)
    elif selected_optim == 'Adam':
        optim = Adam(model.parameters(),
                    lr=cfg['optimizer']['lr'],
                    weight_decay=cfg['optimizer']['weight_decay'])
    else:
        raise NotImplementedError("Optimizer not implemented!")



    # Learning rate schedulers
    warmup_scheduler=None
    lr_scheduler = None
    warmup_period = None
    cosine_scheduler = None

    if cfg['optimizer']['scheduled']:
        print("Executing with scheduled learning rate")
        iter_per_epoch = int((len(train_dataloader) / cfg['training_parameters']['accum_iter']))
        warmup_period = iter_per_epoch * 10
        warmup_scheduler = warmup.LinearWarmup(optim, warmup_period=warmup_period)
        cosine_scheduler = CosineAnnealingLR(optim, T_max=iter_per_epoch*40)



    #################
    # TRAINING LOOP #
    #################

    best_loss = [10000, -1] # loss, epoch
    best_C = [0, -1]        # C, epoch
    epochs = cfg['training_parameters']['epochs']
    accum_iter = cfg['training_parameters']['accum_iter']
    method = cfg['training_parameters']['contrastive_type']
    all_train_loss = []
    all_test_loss = []
    C_indexes = []
    ibs_scores = []

    lrs = []

    for epoch in range(epochs):
        print(f'\nStarting epoch {epoch}')

        if paradigm == 'Cox' or paradigm == 'DeepHit':
            train_loss, epoch_lrs = train_step(model, train_dataloader, optim, loss_fn, 
                                    lr_scheduler=cosine_scheduler, warmup_scheduler=warmup_scheduler, 
                                    warmup_period=warmup_period, device=device)
            test_loss, C, ibs = test_step(model, test_dataloader, loss_fn, device, evaluation='pycox')

        elif cfg['training_parameters']['paradigm'] == 'CoxReg': 
            train_loss, epoch_lrs = train_step_with_reg(model, train_dataloader, optim, loss_fn, 
                                            accum_iter=accum_iter, device=device, 
                                            warmup_scheduler=warmup_scheduler, warmup_period=warmup_period,
                                            lr_scheduler=cosine_scheduler)
            test_loss, C = test_step_with_reg(model, test_dataloader, loss_fn, device)

        elif cfg['training_parameters']['paradigm'] == 'Contrastive':
            train_loss, epoch_lrs = train_step_contrastive(model, train_dataloader, optim, loss_fn, 
                                            accum_iter=accum_iter, device=device, 
                                            warmup_scheduler=warmup_scheduler, warmup_period=warmup_period,
                                            lr_scheduler=cosine_scheduler, method=method)
            test_loss = test_step_contrastive(model, test_dataloader, loss_fn, device, method=method)
            C = 0
            ibs = 0

        # Metrics
        if test_loss < best_loss[0]: 
            best_loss[0] = test_loss
            best_loss[1] = epoch
        if C > best_C[0]: 
            best_C[0] = C
            best_C[1] = epoch
            torch.save(model.state_dict(), f'{exp_id}/best_model.pt')

        all_train_loss.append(train_loss)
        all_test_loss.append(test_loss)
        C_indexes.append(C)
        ibs_scores.append(ibs)
        lrs += epoch_lrs

        print('Train loss:', train_loss)
        print('Test loss :', test_loss)
        print('Test C    :', C)
        print('Test IBS  :', ibs)
        print('LR        :', lrs[-1])

        # Plotting
        plt.figure()
        plt.plot(all_train_loss, label='Train', color='blue')
        plt.plot(all_test_loss, label='Test', color='red')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model loss')
        plt.savefig(f'{exp_id}/loss.png')
        plt.close()

        plt.figure()
        plt.plot(C_indexes, label='C (higher is better)', color='green')
        plt.plot(ibs_scores, label='IBS (lower is better)', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.title('Evaluation metrics')
        plt.savefig(f'{exp_id}/scores.png')
        plt.close()

        plt.figure()
        plt.plot(lrs, label='lr', color='orange')
        plt.xlabel('Iteration')
        plt.ylabel('LR')
        plt.title('Learning rate')
        plt.savefig(f'{exp_id}/lr.png')
        plt.close()


        with open(f'{exp_id}/best.txt', 'w') as f:
            f.write(f'Best Loss: {best_loss[0]} at epoch {best_loss[1]}\n')
            f.write(f'Best C: {best_C[0]} at epoch {best_C[1]}')

        if (epoch + 1) % 25 == 0:
            torch.save(model.state_dict(), f'{exp_id}/e{epoch+1}.pt')


if __name__ == '__main__':

    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        default="config",
                        required=False,
                        help="config")

    arguments = parser.parse_args()

    with open(os.path.join('./'+arguments.config), 'r') as f:
        cfg = yaml.load(f, yaml.Loader)


    train_model(cfg)
