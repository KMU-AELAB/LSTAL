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
from config import *
from data.transform import Cifar

import models.resnet as resnet
import models.featurenet as featurenet
import autoencoder.models.vae as vae

from data.sampler import SubsetSequentialSampler


# Seed
random.seed('KMU_AELAB')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


# Data
transforms = Cifar()

if DATASET == 'cifar10':
    data_train = CIFAR10('./data', train=True, download=True, transform=transforms.train_transform)
    data_unlabeled = CIFAR10('./data', train=True, download=True, transform=transforms.test_transform)
    data_additional = CIFAR10('./data', train=True, download=True, transform=transforms.test_transform)
    data_test = CIFAR10('./data', train=False, download=True, transform=transforms.test_transform)
elif DATASET == 'cifar100':
    data_train = CIFAR100('./data', train=True, download=True, transform=transforms.train_transform)
    data_unlabeled = CIFAR100('./data', train=True, download=True, transform=transforms.test_transform)
    data_additional = CIFAR100('./data', train=True, download=True, transform=transforms.test_transform)
    data_test = CIFAR100('./data', train=False, download=True, transform=transforms.test_transform)


# Train Utils
iters = 0


def additional_train(models, optimizers, dataloaders):
    models['ae'].eval()
    models['backbone'].eval()
    models['module'].train()
    global iters

    _loss, cnt = 0., 0
    for data in tqdm(dataloaders['additional'], leave=False, total=len(dataloaders['additional'])):
        cnt += 1
        iters += 1

        inputs = data[0].cuda()

        optimizers['module'].zero_grad()

        _, features = models['backbone'](inputs)

        features[0] = features[0].detach()
        features[1] = features[1].detach()
        features[2] = features[2].detach()
        features[3] = features[3].detach()

        pred_feature = models['module'](features)
        pred_feature = pred_feature.view([-1, EMBEDDING_DIM])
        ae_out = models['ae'](inputs)

        loss = torch.mean(torch.mean((pred_feature - ae_out[1].detach()) ** 2, dim=1))

        loss.backward()
        optimizers['additional'].step()

        _loss += loss

    return _loss / cnt


def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    models['module'].train()
    models['ae'].eval()
    global iters

    _weight = WEIGHT
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            _weight = WEIGHT / 2
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()

        pred_feature = models['module'](features)
        pred_feature = pred_feature.view([-1, EMBEDDING_DIM])
        ae_out = models['ae'](inputs)

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss = torch.mean(torch.mean((pred_feature - ae_out[1].detach()) ** 2, dim=1))
        loss = m_backbone_loss + _weight * m_module_loss

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()


def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total


def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss):
    print('>> Train a Model.')

    checkpoint_dir = os.path.join(f'./{DATASET}', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        schedulers['module'].step()

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss)

    for epoch in range(num_epochs, num_epochs + (num_epochs // 2)):
        loss = additional_train(models, optimizers, dataloaders)
        schedulers['additional'].step(loss)

    print('>> Finished.')


def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    models['ae'].eval()

    uncertainty = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()

            _, features = models['backbone'](inputs)
            pred_feature = models['module'](features)
            pred_feature = pred_feature.view([-1, EMBEDDING_DIM])

            ae_out = models['ae'](inputs)

            loss = torch.sum((pred_feature - ae_out[1].detach()) ** 2, dim=1)

            uncertainty = torch.cat((uncertainty, loss), 0)
    
    return uncertainty.cpu()


# Main
if __name__ == '__main__':
    for trial in range(TRIALS):
        fp = open(f'record_{trial + 1}.txt', 'w')

        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:INITC]
        unlabeled_set = indices[INITC:]

        train_loader = DataLoader(data_train, batch_size=BATCH,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(data_test, batch_size=BATCH)
        additional_loader = DataLoader(data_additional, batch_size=BATCH,
                                       sampler=SubsetRandomSampler(labeled_set),
                                       pin_memory=True)
        dataloaders = {'train': train_loader, 'test': test_loader, 'additional': additional_loader}

        # Model
        resnet18 = resnet.ResNet18(num_classes=CLS_CNT).cuda()
        feature_module = featurenet.FeatureNet(out_dim=EMBEDDING_DIM).cuda()

        target_module = vae.VAE(NUM_RESIDUAL_LAYERS, NUM_RESIDUAL_HIDDENS, EMBEDDING_DIM)
        checkpoint = torch.load(f'trained_ae/vae.pth.tar')
        target_module.load_state_dict(checkpoint['ae_state_dict'])
        target_module.cuda()

        models = {'backbone': resnet18, 'module': feature_module, 'ae': target_module}

        torch.backends.cudnn.benchmark = False

        # Active learning cycles
        for cycle in range(CYCLES):
            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                                       momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module = optim.SGD(models['module'].parameters(), lr=LR,
                                     momentum=MOMENTUM, weight_decay=WDECAY)
            optim_additional = optim.Adam(models['module'].parameters(), lr=1e-3)

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
            sched_additional = lr_scheduler.ReduceLROnPlateau(optim_additional, mode='min', factor=0.8, cooldown=4)

            optimizers = {'backbone': optim_backbone, 'module': optim_module, 'additional': optim_additional}
            schedulers = {'backbone': sched_backbone, 'module': sched_module, 'additional': sched_additional}

            # Training and test
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL)
            acc = test(models, dataloaders, mode='test')

            fp.write(f'{acc}\n')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(labeled_set), acc))

            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                          sampler=SubsetSequentialSampler(subset),
                                          pin_memory=True)

            # Measure uncertainty of each data points in the subset
            uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order
            arg = np.argsort(uncertainty)

            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)

        # Save a checkpoint
        fp.close()
        torch.save(
            {
            'trial': trial + 1,
            'state_dict_backbone': models['backbone'].state_dict(),
            'state_dict_module': models['module'].state_dict()
            },
            f'./{DATASET}/train/weights/active_resnet18_{DATASET}_trial{trial}.pth'
        )
