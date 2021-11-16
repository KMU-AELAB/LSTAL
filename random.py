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
import models.resnet as resnet
from ll4al_config import *
from data.sampler import SubsetSequentialSampler
from data.transform import Cifar

# Seed
random.seed('KMU_AELAB')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

##
# Data
transforms = Cifar()

if DATASET == 'cifar10':
    data_train = CIFAR10('./data', train=True, download=True, transform=transforms.train_transform)
    data_unlabeled = CIFAR10('./data', train=True, download=True, transform=transforms.test_transform)
    data_test = CIFAR10('./data', train=False, download=True, transform=transforms.test_transform)
elif DATASET == 'cifar100':
    data_train = CIFAR100('./data', train=True, download=True, transform=transforms.train_transform)
    data_unlabeled = CIFAR100('./data', train=True, download=True, transform=transforms.test_transform)
    data_test = CIFAR100('./data', train=False, download=True, transform=transforms.test_transform)


# Train Utils
iters = 0


def train_epoch(model, criterion, opt, dataloaders, epoch):
    model.train()
    global iters

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        opt.zero_grad()

        scores, features = model(inputs)
        target_loss = criterion(scores, labels)

        loss = torch.sum(target_loss) / target_loss.size(0)

        loss.backward()
        opt.step()


def test(model, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _ = model(inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total


def train(model, criterion, opt, scheduler, dataloaders, num_epochs):
    print('>> Train a Model.')
    checkpoint_dir = os.path.join(f'./{DATASET}', 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        scheduler.step()
        train_epoch(model, criterion, opt, dataloaders, epoch)

    print('>> Finished.')


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
        dataloaders = {'train': train_loader, 'test': test_loader}

        # Model
        model = resnet.ResNet18(num_classes=10).cuda()
        torch.backends.cudnn.benchmark = False

        # Active learning cycles
        for cycle in range(CYCLES):
            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')
            opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)

            scheduler = lr_scheduler.MultiStepLR(opt, milestones=MILESTONES)

            # Training and test
            train(model, criterion, opt, scheduler, dataloaders, EPOCH)
            acc = test(model, dataloaders, mode='test')

            fp.write(f'{acc}\n')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(labeled_set), acc))

            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            labeled_set += subset[-ADDENDUM:]
            unlabeled_set = subset[:-ADDENDUM] + unlabeled_set[SUBSET:]

            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)

        # Save a checkpoint
        fp.close()
        torch.save(
            {
            'trial': trial + 1,
            'state_dict_backbone': model.state_dict(),
            },
            f'./{DATASET}/train/weights/active_resnet18_{DATASET}_trial{trial}.pth'
        )
