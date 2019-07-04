import torch
import sys
from sklearn.metrics import balanced_accuracy_score
import numpy as np

def SoftVotingClassifier(probabilities1, probabilities2, probabilities3):
    """ Implements a voting classifier for pre-trained classifiers"""
    probabilities1 = np.array(probabilities1)
    probabilities2 = np.array(probabilities2)
    probabilities3 = np.array(probabilities3)
    preds_ensemble=[]
    for i in range(probabilities1.shape[0]): #aquí et vindrà, per cada imatge, un vector
        list_probs=[]
        for j in range(probabilities1.shape[1]):
            list_probs.append((probabilities1[i][j]+probabilities2[i][j]+probabilities3[i][j])/3)
        print('list_probs',list_probs)
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


model1=sys.argv[1]
model2=sys.argv[2]
model3=sys.argv[3]
#print('se li ha de passar el model sense pth [output_model_my_resnet_18_2018, ]', model)
print('srun --mem 16G python new_ensemble_model_output.py output_model_my_resnet_18_2018 output_model_my_resnet_34_2018 output_model_my_resnet_50_2018')

labels_train_file='model_output/train_labels_'+model1+'.txt' #els labels son tots iguals independentment del model
labels_val_file='model_output/val_labels_'+model1+'.txt'

preds1_train_file='model_output/train_preds_'+model1+'.txt'
preds1_val_file='model_output/val_preds_'+model1+'.txt'

preds2_train_file='model_output/train_preds_'+model2+'.txt'
preds2_val_file='model_output/val_preds_'+model2+'.txt'

preds3_train_file='model_output/train_preds_'+model3+'.txt'
preds3_val_file='model_output/val_preds_'+model3+'.txt'

probs1_train_file='model_output/train_probabilities_'+model1+'.txt'
probs1_val_file='model_output/val_probabilities_'+model1+'.txt'

probs2_train_file='model_output/train_probabilities_'+model2+'.txt'
probs2_val_file='model_output/val_probabilities_'+model2+'.txt'

probs3_train_file='model_output/train_probabilities_'+model3+'.txt'
probs3_val_file='model_output/val_probabilities_'+model3+'.txt'

with open(labels_train_file, 'r') as textFile: #ja esta be
  labels1t = textFile.readlines()
labels_train = [float(s.replace('\n', '')) for s in labels1t]
with open(labels_val_file, 'r') as textFile:
  labels1v = textFile.readlines()
labels_val = [float(s.replace('\n', '')) for s in labels1v]

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

#print('probs1_train',probs1_train)
probs1_train=np.array(probs1_train)
print('probs1_train.shape:',probs1_train.shape[0])
print('probs1_train.shape:',probs1_train.shape[1])
print('probs1_train[8010]',probs1_train[8010])
print('probs1_train[8010]',probs1_train[8010][0])


#preds_ensemble_train=SoftVotingClassifier(probs1_train,probs2_train,probs3_train)
#preds_ensemble_val=SoftVotingClassifier(probs1_val, probs2_val, probs3_val)

preds_ensemble_train=SoftVotingClassifier(probs1_train,0.25,probs2_train,0.25, probs3_train, 0.5)
preds_ensemble_val=SoftVotingClassifier(probs1_val, 0.25, probs2_val, 0.25, probs3_val,0.5 )

balanced_acc_ensemble_train=balanced_accuracy_score(labels_train, preds_ensemble_train, adjusted=True)
print('balanced_acc_ensemble_train',balanced_acc_ensemble_train)

balanced_acc_1_val=balanced_accuracy_score(labels_val, preds1_val, adjusted=True)
print('balanced_1_ensemble_val',balanced_acc_1_val)
balanced_acc_2_val=balanced_accuracy_score(labels_val, preds2_val, adjusted=True)
print('balanced_2_ensemble_val',balanced_acc_2_val)
balanced_acc_3_val=balanced_accuracy_score(labels_val, preds3_val, adjusted=True)
print('balanced_acc_3_val',balanced_acc_3_val)

balanced_acc_ensemble_val=balanced_accuracy_score(labels_val, preds_ensemble_val, adjusted=True)
print('balanced_acc_ensemble_val',balanced_acc_ensemble_val)

