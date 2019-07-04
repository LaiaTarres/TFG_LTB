#aquest fitxer el que preten és carregar el model, extreure'n prediccións, outputs i última capa i anar-ho guardant en fitxers.
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


device = torch.device('cpu')
model_name=sys.argv[1]
print('srun --mem 32G python main_model_to_file.py model_sense_pth any')
print('el model ha de ser model_name sense .pth i tu has posat: ',model_name)
any=sys.argv[2]
print('any:',any)
model=torch.load(model_name+'.pth', map_location=device)
model.eval()

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
} #per normalitzar les imatges.

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
if any=='2018':
    data_dir = '/imatge/ltarres/work/data/ISIC-images/' #aquí posar el del 2018 0 2019
else:
    data_dir = '/imatge/ltarres/work/data/ISIC2019/'
val_dataset = ImageFolderWithPaths(os.path.join(data_dir, 'val'), data_transforms['val'])
train_dataset = ImageFolderWithPaths(os.path.join(data_dir, 'train'), data_transforms['train'])

val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4) #poso batch size de 1 perquè vagi imprimint d'1 en 1
train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4)

escriptura_labels_train = '/imatge/ltarres/PycharmProjects/test/model_output/train_labels_'+model_name+'.txt'
escriptura_labels_val = '/imatge/ltarres/PycharmProjects/test/model_output/val_labels_'+model_name+'.txt'
open(escriptura_labels_train, 'w+').close() #borrem els continguts abans de res
open(escriptura_labels_val, 'w+').close() #borrem els continguts abans de res

#escriptura_outputs_train = '/imatge/ltarres/PycharmProjects/test/model_output/train_output_' + model_name + '.txt'
#escriptura_outputs_val = '/imatge/ltarres/PycharmProjects/test/model_output/val_output_' + model_name + '.txt'
#open(escriptura_outputs_train, 'w+').close()  # borrem els continguts abans de res
#open(escriptura_outputs_val, 'w+').close()  # borrem els continguts abans de res

escriptura_probabilities_train = '/imatge/ltarres/PycharmProjects/test/model_output/train_probabilities_' + model_name + '.txt'
escriptura_probabilities_val = '/imatge/ltarres/PycharmProjects/test/model_output/val_probabilities_' + model_name + '.txt'
open(escriptura_probabilities_train, 'w+').close()  # borrem els continguts abans de res
open(escriptura_probabilities_val, 'w+').close()  # borrem els continguts abans de res

escriptura_preds_train = '/imatge/ltarres/PycharmProjects/test/model_output/train_preds_'+model_name+'.txt'
escriptura_preds_val = '/imatge/ltarres/PycharmProjects/test/model_output/val_preds_'+model_name+'.txt'
open(escriptura_preds_train, 'w+').close() #borrem els continguts abans de res
open(escriptura_preds_val, 'w+').close() #borrem els continguts abans de res

escriptura_features_train = '/imatge/ltarres/PycharmProjects/test/model_output/train_params_'+model_name+'.txt'
escriptura_features_val = '/imatge/ltarres/PycharmProjects/test/model_output/val_params_'+model_name+'.txt'
open(escriptura_features_train, 'w+').close() #borrem els continguts abans de res
open(escriptura_features_val, 'w+').close() #borrem els continguts abans de res

count=0

for phase in ['train', 'val']:
    if phase == 'train':
        for inputs, labels, paths in train_dataloader: #ho fem imatge a imatge
            count+=1
            print('imatge numero(train): ',count)
            #print('labels.data:',labels.data) #tensor[0]
            #print('labels.data.item():', labels.data.item())  # tensor[0]
            with open(escriptura_labels_train, 'a+') as f:
                f.write('{}\n'.format(labels.data.item())) #extraiem el ground truth

            outputs = model(inputs) #tensor([[ 3.8025,  2.5423, -0.0646, -1.9939, -1.8498, -0.6969, -1.6206]],grad_fn=<AddmmBackward>)
            #print('outputs:',outputs) #extraiem el vector amb les probabilitats
            #print('outputs.data.tolist():', outputs.tolist())
            #with open(escriptura_outputs_train, 'a+') as g:
            #    g.write('{}\n'.format(outputs.tolist()))

            sm = torch.nn.Softmax()
            probabilities = sm(outputs) #tensor([[0.7538, 0.2138, 0.0158, 0.0023, 0.0026, 0.0084, 0.0033]], grad_fn=<SoftmaxBackward>)
            #print('probabilities',probabilities)
            #print('probabilities.tolist()', probabilities.tolist())
            with open(escriptura_probabilities_train, 'a+') as h:
                h.write('{}\n'.format(probabilities.tolist()))

            _, preds = torch.max(outputs, 1) #extraiem la prediccio
            #print('preds.data',preds.data) #tensor([0])
            #print('preds.data.item()', preds.data.item())
            with open(escriptura_preds_train, 'a+') as i:
                i.write('{}\n'.format(preds.data.item()))

            features=model.extract_features(inputs)#tensor([[7.3523e-01,....1.5753e+00]], grad_fn=<ViewBackward>) #el vector de dins té 6x85+2=512
            #print('features', features)
            #print('features.tolist()', features.tolist())
            with open(escriptura_features_train, 'a+') as j:
                j.write('{}\n'.format(features.tolist()))
    else:
        for inputs, labels, paths in val_dataloader:#ho fem imatge a imatge
            count+=1
            print('imatge numero (val): ', count)
            #print('labels.data:', labels.data)  # tensor[0]
            # print('labels.data.item():', labels.data.item())  # tensor[0]
            with open(escriptura_labels_val, 'a+') as f:
                f.write('{}\n'.format(labels.data.item()))  # extraiem el ground truth

            outputs = model(
                inputs)  # tensor([[ 3.8025,  2.5423, -0.0646, -1.9939, -1.8498, -0.6969, -1.6206]],grad_fn=<AddmmBackward>)
            #print('outputs:', outputs)  # extraiem el vector amb les probabilitats
            # print('outputs.data.tolist():', outputs.tolist())
            #with open(escriptura_outputs_val, 'a+') as g:
            #    g.write('{}\n'.format(outputs.tolist()))

            sm = torch.nn.Softmax()
            probabilities = sm(
                outputs)  # tensor([[0.7538, 0.2138, 0.0158, 0.0023, 0.0026, 0.0084, 0.0033]], grad_fn=<SoftmaxBackward>)
            #print('probabilities_val', probabilities)
            #print('probabilities_val.tolist()', probabilities.tolist())
            with open(escriptura_probabilities_val, 'a+') as h:
                h.write('{}\n'.format(probabilities.tolist()))

            _, preds = torch.max(outputs, 1)  # extraiem la prediccio
            #print('preds.data', preds.data)  # tensor([0])
            # print('preds.data.item()', preds.data.item())
            with open(escriptura_preds_val, 'a+') as i:
                i.write('{}\n'.format(preds.data.item()))

            features = model.extract_features(inputs)  # tensor([[7.3523e-01,....1.5753e+00]], grad_fn=<ViewBackward>) #el vector de dins té 6x85+2=512
            #print('features', features)
            # print('features.tolist()', features.tolist())
            with open(escriptura_features_val, 'a+') as j:
                j.write('{}\n'.format(features.tolist()))

#per executar srun --mem 16G python main_model_to_file.py model_sense_pth any
# srun --mem 16G python main_model_to_file.py output_model_my_resnet_18_2019_ub 2019
