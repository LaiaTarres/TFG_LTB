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

argument_list=str(sys.argv)
print('argument_list: [model1,model2,model3]',argument_list)
print('for example: [output_model_resnet18_100ep_lr6_bs36_balanced.pth,output_model_resnet18_70ep_lr6_bs36_no_balanced.pth,output_model_resnet18_70ep_lr6_bs36_balanced.pth]')
print('Ensemble per: ['+sys.argv[1]+','+sys.argv[2]+','+sys.argv[3]+']')
print('batch_size=',36)
batch_size=36


def hardVotingClassifier(preds_model1, preds_model2, preds_model3):
    """Se li passen els 4 tensors amb la info acumulada. S'ha de fer la moda de cada imatge"""
    hard_voting_preds=[]
    for i in range(len(preds_model1)):
        list_preds = [preds_model1.data[i], preds_model2.data[i], preds_model3.data[i]]#llista amb les tres posicions
        #hauria de fer un if, que si no hi ha una moda, que es quedi amb el primer
        hard_voting_preds.append(max(set(list_preds), key=list_preds.count))  # Returns the highest occurring item
        #creo una llista amb la  posició i de cada model
        #hauria d'estar amb el tipus que el d'entrada

    return torch.tensor(hard_voting_preds, dtype=torch.long, device=device)

def SoftVotingClassifier(outputs1, outputs2, outputs3):
    """ Implements a voting classifier for pre-trained classifiers"""
    for i in range(len(outputs1)): #aquí et vindrà, per cada imatge del batch, un vector
        #tensor([[[2.5403, 1.0738, -2.0716, 0.7750, -1.1602, 1.2238, -1.8473],...], grad_fn=...)
        #list_outputs=[outputs1.data[i],outputs2.data[i],outputs3.data[i]] #això és un vector de 3 posicions, a cadascuna hi ha le sprobabilitats de predicció de cada classe... però les probs son rares
        list_outputs[j]=sum(outputs1.data[i](j),outputs2.data[i](j),outputs3.data[i](j))/3 #Agafes la classe j
        return pos(max(list_outputs)) #retornes la posició on s'ha trobat el màxim (0..6)


#def WeightedVotingClassifier(object):
    #""" Implements a voting classifier for pre-trained classifiers"""

#carreguem els tres models per separat
device = torch.device('cpu')

model1=torch.load(sys.argv[1], map_location=device)
model1.eval()

model2=torch.load(sys.argv[2], map_location=device)
model2.eval()

model3=torch.load(sys.argv[3], map_location=device)
model3.eval()

#ho evaluem per veure cadascun els outputs que ha tret. Però tot s'ha de fer el local, no vull posar res a la gpu

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

data_dir = '/imatge/ltarres/work/data/ISIC-images/'
val_dataset = ImageFolderWithPaths(os.path.join(data_dir, 'val'), data_transforms['val'])
train_dataset = ImageFolderWithPaths(os.path.join(data_dir, 'train'), data_transforms['train'])

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4) #he canviat num_workers... sembla que no tinc memòria, em diu killed by signal però continua sortint el matiex error
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)

data_acc = torch.tensor([], dtype=torch.long, device=device)
preds_acc1 = torch.tensor([], dtype=torch.long, device=device)
preds_acc2 = torch.tensor([], dtype=torch.long, device=device)
preds_acc3 = torch.tensor([], dtype=torch.long, device=device)
preds_acc_ensemble_train = torch.tensor([], dtype=torch.long, device=device)
preds_acc_ensemble_val = torch.tensor([], dtype=torch.long, device=device)

#running_loss_train=0.0
#running_loss_val=0.0

escriptura_acc_train = '/imatge/ltarres/PycharmProjects/test/loss_acc/train_acc_ensemble.txt'
escriptura_acc_val = '/imatge/ltarres/PycharmProjects/test/loss_acc/val_acc_ensemble.txt'
open(escriptura_acc_train, 'w').close() #borrem els continguts abans de res
open(escriptura_acc_val, 'w').close() #borrem els continguts abans de res

for phase in ['train', 'val']:
    if phase == 'train':
        for inputs, labels, paths in train_dataloader:
            #inputs = inputs.to(device) #potser el to(device) no cal...
            #labels = labels.to(device)
            data_acc = torch.cat((data_acc, labels.data), 0)
            print('labels:', labels)
            outputs1 = model1(inputs)
            sm = torch.nn.Softmax()
            probabilities1 = sm(outputs1)
            _, preds1 = torch.max(outputs1, 1) #tensor([4,4....,4], device='cuda:0')
            #print('outputs train model 1:',outputs1)
            print('preds train model1: ', preds1)
            preds_acc1 = torch.cat((preds_acc1, preds1), 0)

            outputs2 = model2(inputs)
            sm = torch.nn.Softmax()
            probabilities2 = sm(outputs2)
            _, preds2 = torch.max(outputs2, 1)  # tensor([4,4....,4], device='cuda:0')
            #print('outputs train model 2:', outputs2)
            print('preds train model2: ', preds2)
            preds_acc2 = torch.cat((preds_acc2, preds2), 0)

            outputs3 = model3(inputs)
            _, preds3 = torch.max(outputs3, 1)  # tensor([4,4....,4], device='cuda:0')
            #print('outputs train model 3:', outputs3)
            sm = torch.nn.Softmax()
            probabilities = sm(outputs3)
            print('preds train model3: ', preds3)
            preds_acc3 = torch.cat((preds_acc3, preds3), 0)

            preds_ensemble=hardVotingClassifier(preds1,preds2,preds3)
            print('preds_ensemble:',preds_ensemble)
            preds_acc_ensemble_train=torch.cat((preds_acc_ensemble_train, preds_ensemble), 0)

            #loss = criterion(outputs, labels)
            #running_loss_train += loss.item() * inputs.size(0)
        epoch_balanced_acc = balanced_accuracy_score(data_acc, preds_acc_ensemble_train, adjusted=True)
        print('epoch_balanced_acc',epoch_balanced_acc)
        with open(escriptura_acc_train, 'a+') as f:
            f.write('{}\n'.format(epoch_balanced_acc))

    else:
        for inputs, labels, paths in val_dataloader:
            #inputs = inputs.to(device)
            #labels = labels.to(device)
            data_acc = torch.cat((data_acc, labels.data), 0)
            outputs1 = model1(inputs)
            print('labels val:', labels)
            _, preds1 = torch.max(outputs1, 1)  # tensor([4,4....,4], device='cuda:0')
            print('preds val model1: ', preds1)
            preds_acc1 = torch.cat((preds_acc1, preds1), 0)

            outputs2 = model2(inputs)
            _, preds2 = torch.max(outputs2, 1)  # tensor([4,4....,4], device='cuda:0')
            print('preds val model2: ', preds2)
            preds_acc2 = torch.cat((preds_acc2, preds2), 0)

            outputs3 = model3(inputs)
            _, preds3 = torch.max(outputs3, 1)  # tensor([4,4....,4], device='cuda:0')
            print('preds val model3: ', preds3)
            preds_acc3 = torch.cat((preds_acc3, preds3), 0)

            preds_ensemble = hardVotingClassifier(preds1, preds2, preds3)
            print('preds_ensemble:', preds_ensemble)
            preds_acc_ensemble_val = torch.cat((preds_acc_ensemble_val, preds_ensemble), 0)
            #running_loss_val += loss.item() * inputs.size(0)
        #epoch_loss = running_loss / dataset_sizes[phase]
        epoch_balanced_acc = balanced_accuracy_score(data_acc, preds_acc_ensemble_val, adjusted=True)
        print('epoch_balanced_acc', epoch_balanced_acc)
        with open(escriptura_acc_val, 'a+') as f:
            f.write('{}\n'.format(epoch_balanced_acc))


#per executar: srun --mem 32G python main_ensemble.py output_model_resnet18_100ep_lr6_bs36_balanced.pth output_model_resnet18_70ep_lr6_bs36_no_balanced.pth output_model_resnet18_70ep_lr6_bs36_balanced.pth


##guardr les prediccions a un lloc
##grid search de las predicciones (alpha)

##preguntar la tere per les features
#resnet 18: prediccions, probabilitats, última capa de features
#resnet 34:
#resnet 50:
#predicciones max voting,
#probabilitats: amb soft voting
#amb la última capa de features, entrenar 3 clasificadores (random forests o Ada Boost)

#para cada red, sale un vector de características distinto. Luego, con la salida de cada red entrenar una svm o otra cosa, luego combinarlos y hacer un voting
#sino per cada red, tres SVM i després utilitzar el mètode de sickit learn de voting per tenir un millor
#

