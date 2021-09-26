'''Get dataset mean and std with PyTorch.'''


from copy import deepcopy
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from options import Options
# import options as Options
from lib.data import load_data
# from lib.models.ganomaly import Ganomaly
# from lib.models.f_anogan import F_anogan
from tqdm import tqdm
import json

import os
import argparse
import numpy as np
# import models
import time
from torchvision import transforms, datasets


def mean__std(data_loader):
    cnt = 0
    mean = torch.empty(3)
    std = torch.empty(3)

    for data, _, _ in data_loader:

        b, c, h, w = data.shape
        
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * mean + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * std + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return mean, torch.sqrt(std - mean ** 2)


if __name__ == '__main__':
        
    opt = Options().parse()
    # Data
    print('==> Preparing data..')
    train_dataset =  load_data(opt)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h, w = 0, 0

    ###
    # generate_batches()
    # for batch_idx, (inputs) in enumerate(training_loader):
    for batch_idx, data in enumerate(train_dataset['train']):
        inputs = data[0]
        
        
        # inputs = inputs[0].to(device)
        inputs = inputs.to(device)
        # print(inputs.size(0))
        # exit()
        if batch_idx == 0:
            h, w = inputs.size(2), inputs.size(3)
            print(inputs.min(), inputs.max())
            chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
            #chsum = inputs.sum(dim=0, keepdim=True)
        else:
            chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
            #chsum += inputs.sum(dim=0, keepdim=True)
    mean = chsum/len(train_dataset)/h/w
    print('mean: %s' % mean.view(-1))
    
    chsum = None

    for batch_idx, data in enumerate(train_dataset['train']):
        inputs = data[0].to(device)
        if batch_idx == 0:
            chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
            #chsum = inputs.sum(dim=0, keepdim=True)
        else:
            chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
            #chsum += inputs.sum(dim=0, keepdim=True)
    std = torch.sqrt(chsum/(len(train_dataset) * h * w - 1))
    print('std: %s' % std.view(-1))

    print('Done!')