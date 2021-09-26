"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from lib.networks import NetG, NetD, weights_init
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import evaluate

from lib.model import *

from options import Options
from lib.data import load_data
from lib.model import Ganomaly
#from lib.casting_dataset import *
import torch

##
def test():
    """ Training
    """

    ##
    # ARGUMENTS
    print("set option")
    opt = Options().parse()
    ##
    # LOAD DATA
    print("get data")
    #dataset = Castingdataset(root='data', train=True)
    ##
    # LOAD MODEL
    print('set model')
    dataloader = load_data(opt)
    model = Ganomaly(opt, dataloader)
    ##
    # TRAIN MODEL
    print('start test')
    result = model.test()
    print(result)

if __name__ == '__main__':
    test()