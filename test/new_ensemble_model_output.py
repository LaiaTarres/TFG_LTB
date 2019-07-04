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
from sklearn import svm


#preten fer els 3 ensembles diferents a partir dels fitxers de la carpeta test/model_output

#primer definim els tipus d'ensembles diferents
def hardVotingClassifier(preds_model1, preds_model2, preds_model3):
    """Se li passen els 4 tensors amb la info acumulada. S'ha de fer la moda de cada imatge"""
    hard_voting_preds=[]
    for i in range(len(preds_model1)): #range perquè li estavem tensor amb un vector de mida batch size, aquí ja li passarem un vector
        list_preds = [preds_model1[i], preds_model2[i], preds_model3[i]]#llista amb les tres posicions
        hard_voting_preds.append(max(set(list_preds), key=list_preds.count))  # Returns the highest occurring item

    return hard_voting_preds#torch.tensor(hard_voting_preds, dtype=torch.long, device=device)

def softVotingClassifier(probabilities1, probabilities2, probabilities3):
    """ Implements a voting classifier for pre-trained classifiers"""
    probabilities1 = np.array(probabilities1)
    probabilities2 = np.array(probabilities2)
    probabilities3 = np.array(probabilities3)
    preds_ensemble=[]
    for i in range(probabilities1.shape[0]): #aquí et vindrà, per cada imatge, un vector
        list_probs=[]
        for j in range(probabilities1.shape[1]):
            list_probs.append((probabilities1[i][j]+probabilities2[i][j]+probabilities3[i][j])/3)
        #print('list_probs',list_probs)
        preds_ensemble.append(list_probs.index(max(list_probs))) #et torna la posició del màxim, s'ha de comprovar que vagi de 0 a 6

    return preds_ensemble

def softVotingClassifierWeighted(probabilities1, w1, probabilities2, w2, probabilities3, w3):
    """ Implements a voting classifier for pre-trained classifiers"""
    probabilities1 = np.array(probabilities1)
    probabilities2 = np.array(probabilities2)
    probabilities3 = np.array(probabilities3)
    preds_ensemble=[]
    for i in range(probabilities1.shape[0]): #aquí et vindrà, per cada imatge, un vector
        list_probs=[]
        for j in range(probabilities1.shape[1]):
            list_probs.append((w1*probabilities1[i][j]+w2*probabilities2[i][j]+w3*probabilities3[i][j])/3)
        #print('list_probs',list_probs)
        preds_ensemble.append(list_probs.index(max(list_probs))) #et torna la posició del màxim, s'ha de comprovar que vagi de 0 a 6

    return preds_ensemble

def svm_prediction(params_train, labels_train, params_val):
    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel, posar les caracteristiques
    # Train the model using the training sets
    clf.fit(params_train, labels_train)
    # Predict the response for val dataset
    label_pred = clf.predict(params_val) #volem que això sigui una llista
    return label_pred.tolist() #retornes les prediccions del svm d'aquell model


#després llegim els fitxers
model=sys.argv[1] #aixo s'ha de treure...

model1=sys.argv[1]
model2=sys.argv[2]
model3=sys.argv[3]
tipus_ensemble=sys.argv[4]
print('se li ha de passar el model sense pth []', model1, model2, model3, tipus_ensemble)
#print('srun --mem 16G python new_ensemble_model_output.py output_model_my_resnet_18_2018 output_model_my_resnet_34_2018 output_model_my_resnet_50_2018 svm_hard_voting')

#labels_train_file='model_output/train_labels_'+model+'.txt'
#output_train_file='model_output/train_output_'+model+'.txt'
#params_train_file='model_output/train_params_'+model+'.txt'
#preds_train_file='model_output/train_preds_'+model+'.txt'
#probs_train_file='model_output/train_probabilities_'+model+'.txt'

#labels_val_file='model_output/val_labels_'+model+'.txt'
#output_val_file='model_output/val_output_'+model+'.txt'
#params_val_file='model_output/val_params_'+model+'.txt'
#preds_val_file='model_output/val_preds_'+model+'.txt'
#probs_val_file='model_output/val_probabilities_'+model+'.txt'

labels_train_file='model_output/train_labels_'+model1+'.txt' #els labels son tots iguals independentment del model
labels_val_file='model_output/val_labels_'+model1+'.txt'

params1_train_file='model_output/train_params_'+model1+'.txt'
params1_val_file='model_output/val_params_'+model1+'.txt'

params2_train_file='model_output/train_params_'+model2+'.txt'
params2_val_file='model_output/val_params_'+model2+'.txt'

params3_train_file='model_output/train_params_'+model3+'.txt'
params3_val_file='model_output/val_params_'+model3+'.txt'

probs1_train_file='model_output/train_probabilities_'+model1+'.txt'
probs1_val_file='model_output/val_probabilities_'+model1+'.txt'

probs2_train_file='model_output/train_probabilities_'+model2+'.txt'
probs2_val_file='model_output/val_probabilities_'+model2+'.txt'

probs3_train_file='model_output/train_probabilities_'+model3+'.txt'
probs3_val_file='model_output/val_probabilities_'+model3+'.txt'

preds1_train_file='model_output/train_preds_'+model1+'.txt'
preds1_val_file='model_output/val_preds_'+model1+'.txt'

preds2_train_file='model_output/train_preds_'+model2+'.txt'
preds2_val_file='model_output/val_preds_'+model2+'.txt'

preds3_train_file='model_output/train_preds_'+model3+'.txt'
preds3_val_file='model_output/val_preds_'+model3+'.txt'


with open(labels_train_file, 'r') as textFile: #ja esta be
  labels1t = textFile.readlines()
labels_train = [float(s.replace('\n', '')) for s in labels1t]
with open(labels_val_file, 'r') as textFile:
  labels1v = textFile.readlines()
labels_val = [float(s.replace('\n', '')) for s in labels1v]


with open(params1_train_file, 'r') as textFile:
    params1t = textFile.readlines()
params1_train = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in params1t]
with open(params1_val_file, 'r') as textFile:
    params1v = textFile.readlines()
params1_val = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in params1v]

with open(params2_train_file, 'r') as textFile:
    params1t = textFile.readlines()
params2_train = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in params1t]
with open(params2_val_file, 'r') as textFile:
    params1v = textFile.readlines()
params2_val = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in params1v]

with open(params3_train_file, 'r') as textFile:
    params1t = textFile.readlines()
params3_train = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in params1t]
with open(params3_val_file, 'r') as textFile:
    params1v = textFile.readlines()
params3_val = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in params1v]

with open(preds1_train_file, 'r') as textFile: #ja està bé
  preds1t = textFile.readlines()
preds1_train = [float(s.replace('\n', '')) for s in preds1t]
with open(preds1_val_file, 'r') as textFile:
  preds1v = textFile.readlines()
preds1_val = [float(s.replace('\n', '')) for s in preds1v]

with open(preds2_train_file, 'r') as textFile: #ja està bé
  preds1t = textFile.readlines()
preds2_train = [float(s.replace('\n', '')) for s in preds1t]
with open(preds2_val_file, 'r') as textFile:
  preds1v = textFile.readlines()
preds2_val = [float(s.replace('\n', '')) for s in preds1v]

with open(preds3_train_file, 'r') as textFile: #ja està bé
  preds1t = textFile.readlines()
preds3_train = [float(s.replace('\n', '')) for s in preds1t]
with open(preds3_val_file, 'r') as textFile:
  preds1v = textFile.readlines()
preds3_val = [float(s.replace('\n', '')) for s in preds1v]

with open(probs1_train_file, 'r') as textFile:
  probs1t = textFile.readlines()
probs1_train = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in probs1t]
with open(probs1_val_file, 'r') as textFile:
  probs1v = textFile.readlines()
probs1_val = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in probs1v]

with open(probs2_train_file, 'r') as textFile:
  probs1t = textFile.readlines()
probs2_train = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in probs1t]
with open(probs2_val_file, 'r') as textFile:
  probs1v = textFile.readlines()
probs2_val = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in probs1v]

with open(probs3_train_file, 'r') as textFile:
  probs1t = textFile.readlines()
probs3_train = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in probs1t]
with open(probs3_val_file, 'r') as textFile:
  probs1v = textFile.readlines()
probs3_val = [[float(num) for num in s.replace('\n', '').replace('[[','').replace(']]','').split(',')] for s in probs1v]

#fem print de tot per comprovar que ho estigui fent bé


#una vegada ho tenim tot en vectors, ho passem als diferents ensembles per obtenir uns outputs diferents
if tipus_ensemble=='hard_voting':
    preds_ensemble_train = hardVotingClassifier(preds1_train, preds2_train, preds3_train)
    preds_ensemble_val = hardVotingClassifier(preds1_val, preds2_val, preds3_val)
elif tipus_ensemble=='soft_voting':
    preds_ensemble_train = softVotingClassifier(probs1_train, probs2_train, probs3_train)
    preds_ensemble_val = softVotingClassifier(probs1_val, probs2_val, probs3_val)
elif tipus_ensemble=='soft_voting_weighted':
    w1=0.5#0.7735238186475607/(0.7735238186475607+0.745933765748567+0.7585784365027333) #0.5
    w2=0.25#0.745933765748567/(0.7735238186475607+0.745933765748567+0.7585784365027333) #0.25
    w3=0.25#0.7585784365027333/(0.7735238186475607+0.745933765748567+0.7585784365027333) #0.25
    preds_ensemble_train=softVotingClassifierWeighted(probs1_train, w1, probs2_train, w2, probs3_train, w3)
    preds_ensemble_val=softVotingClassifierWeighted(probs1_val, w1, probs2_val, w2, probs3_val, w3)
elif tipus_ensemble=='svm_hard_voting':
    preds_svm_val_1 = svm_prediction(params1_train, labels_train)
    preds_svm_val_2 = svm_prediction(params2_train, labels_train)
    preds_svm_val_3 = svm_prediction(params3_train, labels_train)
    preds_ensemble_val = hardVotingClassifier(preds_svm_val_1, preds_svm_val_2, preds_svm_val_3)
else :
    print('El tipo de ensemble no se ha definido correctamente')

#veiem com ha millorat l'accuracy amb l'ensemble
balanced_acc_1_train=balanced_accuracy_score(labels_train, preds1_train, adjusted=True)
print('balanced_acc_1_train',balanced_acc_1_train)
balanced_acc_2_train=balanced_accuracy_score(labels_train, preds2_train, adjusted=True)
print('balanced_acc_2_train',balanced_acc_2_train)
balanced_acc_3_train=balanced_accuracy_score(labels_train, preds3_train, adjusted=True)
print('balanced_acc_3_train',balanced_acc_3_train)

balanced_acc_ensemble_train=balanced_accuracy_score(labels_train, preds_ensemble_train, adjusted=True)
print('balanced_acc_ensemble_train',balanced_acc_ensemble_train)

balanced_acc_1_val=balanced_accuracy_score(labels_val, preds1_val, adjusted=True)
print('balanced_acc_1_val',balanced_acc_1_val)
balanced_acc_2_val=balanced_accuracy_score(labels_val, preds2_val, adjusted=True)
print('balanced_acc_2_val',balanced_acc_2_val)
balanced_acc_3_val=balanced_accuracy_score(labels_val, preds3_val, adjusted=True)
print('balanced_acc_3_val',balanced_acc_3_val)

balanced_acc_ensemble_val=balanced_accuracy_score(labels_val, preds_ensemble_val, adjusted=True)
print('balanced_acc_ensemble_val',balanced_acc_ensemble_val)