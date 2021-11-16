'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

# Python
import os
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
from torchvision.datasets import CIFAR100, CIFAR10

# Utils
from tqdm import tqdm

# Custom
from models.vae import VAE
from config import *
from transform import Cifar

# Seed
random.seed('KMU_AELAB')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

##
# Data
transforms = Cifar()

if DATASET == 'cifar10':
    data_train = CIFAR10('../data', train=True, download=True, transform=transforms.transform)
    data_unlabeled = CIFAR10('../data', train=True, download=True, transform=transforms.transform)
    data_test = CIFAR10('../data', train=False, download=True, transform=transforms.transform)
elif DATASET == 'cifar100':
    data_train = CIFAR100('../data', train=True, download=True, transform=transforms.transform)
    data_unlabeled = CIFAR100('../data', train=True, download=True, transform=transforms.transform)
    data_test = CIFAR100('../data', train=False, download=True, transform=transforms.transform)


# Train Utils
iters = 0


def train_epoch(model, criterion, opt, dataloaders):
    model.train()
    global iters

    cnt = 0
    _loss = 0.
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        cnt += 1
        iters += 1

        opt.zero_grad()
        inputs = data[0].cuda()

        recon, features, mu, logvar = model(inputs)
        recon_loss = criterion(recon, inputs)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        loss = recon_loss + (0.001 * kld_loss)

        _loss += loss

        loss.backward()
        opt.step()

    return _loss / cnt


def test(model, criterion, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    model.eval()

    loss = 0.
    with torch.no_grad():
        for (inputs, _) in dataloaders[mode]:
            inputs = inputs.cuda()

            recon, _, _, _ = model(inputs)
            loss += criterion(recon, inputs)

    return loss


def train(model, criterion, opt, scheduler, dataloaders, num_epochs, trial):
    print('>> Train a Model.')
    best_loss = 999999999.
    checkpoint_dir = os.path.join(f'./{DATASET}', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        loss = train_epoch(model, criterion, opt, dataloaders)
        scheduler.step(loss)

        # Save a checkpoint
        if epoch % 5 == 4:
            _loss = test(model, criterion, dataloaders, 'test')
            if best_loss > _loss:
                best_loss = _loss
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'ae_state_dict': model.state_dict(),
                    },
                    f'../trained_vae/ae_{trial}.pth'
                )
            print('Val loss: {:.3f} \t Best loss: {:.3f}'.format(_loss, best_loss))
    print('>> Finished.')


# Main
if __name__ == '__main__':
    for trial in range(TRIALS):
        train_loader = DataLoader(data_train, batch_size=BATCH, pin_memory=True)
        test_loader = DataLoader(data_test, batch_size=BATCH)
        dataloaders = {'train': train_loader, 'test': test_loader}

        # Model
        model = VAE(NUM_RESIDUAL_LAYERS, NUM_RESIDUAL_HIDDENS, EMBEDDING_DIM).cuda()
        torch.backends.cudnn.benchmark = False

        # Loss, criterion and scheduler (re)initialization
        criterion = nn.MSELoss()
        opt = optim.Adam(model.parameters(), lr=LR)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.8, cooldown=4)

        # Training and test
        train(model, criterion, opt, scheduler, dataloaders, EPOCH, trial)
