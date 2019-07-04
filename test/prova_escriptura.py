from __future__ import print_function, division

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

src_dir = str('/imatge/ltarres/work/data/ISIC-images/val/nevus/ISIC_0026183.jpg')
dst_dir_aux=src_dir.split('/val/') #separar-lo
print('dst_dir_aux', dst_dir_aux)
print('dst_dir_aux[0]', dst_dir_aux[0])
dst_dir_aux_2=dst_dir_aux[1].split('/')
print('dst_dir_aux_2[1]', dst_dir_aux_2[1])
dst_dir = dst_dir_aux[0] +'/output/' + 'nevus' + '/' + dst_dir_aux_2[1]
print('src_dir',src_dir)
print('dst_dir',dst_dir)
