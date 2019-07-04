from __future__ import print_function, division

import numpy as np
from sklearn.metrics import confusion_matrix

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
import glob
import shutil
import sys
from sklearn.metrics import balanced_accuracy_score

output_train_file='train_output_'+'output_model_my_resnet_18_2018'+'.txt'
print('output_train_file:',output_train_file)
#with open(output_train_file, 'r') as textFile:
  #output1t = textFile.readlines()
#output_train = [s.replace('\n', '').replace('[[','').replace(']]','').split(',') for s in output1t] #aquí ho tens tot super gran

#for i in output_train:
#    print('i:', i)
#    for j in i:
#        print('float(j)',float(j))

with open(output_train_file, 'r') as textFile:
  output1t = textFile.readlines()
output_train=[[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in output1t]
print('output_train: ',output_train)

x=np.array(output_train)
print('x.shape:',x.shape)
print('x[8010]',x[8010]) #això és tot el vector
print('x[8010][0]',x[8010][0]) #això és la primera posició


