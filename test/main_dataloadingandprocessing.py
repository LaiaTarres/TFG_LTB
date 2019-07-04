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
from sklearn.metrics import balanced_accuracy_score

escriptura_loss_val = '/imatge/ltarres/PycharmProjects/test/loss_acc/val_loss_lr6_bs36_balancederr.txt'
escriptura_loss_train = '/imatge/ltarres/PycharmProjects/test/loss_acc/train_loss_lr6_bs36_balancederr.txt'
escriptura_acc_train = '/imatge/ltarres/PycharmProjects/test/loss_acc/train_acc_lr6_bs36_balancederr.txt'
escriptura_acc_val = '/imatge/ltarres/PycharmProjects/test/loss_acc/val_acc_lr6_bs36_balancederr.txt'
open(escriptura_loss_val, 'w').close() #borrem els continguts abans de res
open(escriptura_loss_train, 'w').close() #borrem els continguts abans de res
open(escriptura_acc_train, 'w').close() #borrem els continguts abans de res
open(escriptura_acc_val, 'w').close() #borrem els continguts abans de res
