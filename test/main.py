#!/usr/bin/python
from __future__ import print_function, division

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

argument_list=str(sys.argv)
print('argument_list: [model,epoch,learning_rate,batch_size]',argument_list)
print('for example: [resnet18, 100, 6, 36]')
model=sys.argv[1]
epoch=sys.argv[2]
learning_rate='1e-'+str(sys.argv[3])
batch_size=sys.argv[4]
print('learning_rate:',learning_rate)



#per script_run_no_balanced.sh
#python main_err_acc_balanced.py resnet18 100 6 36 && python /loss_acc/plot_acc.py resnet18 100 6 36 && python /loss_acc/plot_loss.py resnet18 100 6 36 && python main_confusion_matrix_proves.py resnet18 100 6 36 n