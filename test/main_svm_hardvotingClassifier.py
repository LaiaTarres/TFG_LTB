import sys
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn import svm

def hardVotingClassifier(preds_model1, preds_model2, preds_model3):
    """Se li passen els 4 tensors amb la info acumulada. S'ha de fer la moda de cada imatge"""
    hard_voting_preds=[]
    for i in range(len(preds_model1)): #range perquè li estavem tensor amb un vector de mida batch size, aquí ja li passarem un vector
        list_preds = [preds_model1[i], preds_model2[i], preds_model3[i]]#llista amb les tres posicions
        hard_voting_preds.append(max(set(list_preds), key=list_preds.count))  # Returns the highest occurring item

    return hard_voting_preds#torch.tensor(hard_voting_preds, dtype=torch.long, device=device)

def svm_prediction(params_train, labels_train, params_val):
    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel, posar les caracteristiques
    # Train the model using the training sets
    clf.fit(params_train, labels_train)
    # Predict the response for val dataset
    label_pred = clf.predict(params_val) #volem que això sigui una llista
    return label_pred.tolist() #retornes les prediccions del svm d'aquell model



model1=sys.argv[1]
model2=sys.argv[2]
model3=sys.argv[3]
print('se li ha de passar el model sense pth [output_model_my_resnet_18_2018 output_model_my_resnet_34_2018 output_model_my_resnet_51_2018]', model1, model2, model3)
print('srun --mem 16G python new_ensemble_model_output.py output_model_my_resnet_18_2018 output_model_my_resnet_34_2018 output_model_my_resnet_50_2018')

labels_train_file='model_output/train_labels_'+model1+'.txt' #els labels son tots iguals independentment del model
labels_val_file='model_output/val_labels_'+model1+'.txt'

params1_train_file='model_output/train_params_'+model1+'.txt'
params1_val_file='model_output/val_params_'+model1+'.txt'

params2_train_file='model_output/train_params_'+model2+'.txt'
params2_val_file='model_output/val_params_'+model2+'.txt'

params3_train_file='model_output/train_params_'+model3+'.txt'
params3_val_file='model_output/val_params_'+model3+'.txt'

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

#una vegada llegits, ja podem executar
preds_svm_val_1=svm_prediction(params1_train, labels_train, params1_val)
print('preds_svm_val_1 done')
preds_svm_val_2=svm_prediction(params2_train, labels_train, params2_val)
print('preds_svm_val_2 done')
preds_svm_val_3=svm_prediction(params3_train, labels_train, params3_val) #el problema el dona aquí, com si params1_val
print('preds_svm_val_3 done')
preds_ensemble_val = hardVotingClassifier(preds_svm_val_1, preds_svm_val_2, preds_svm_val_3)
print('preds_ensemble_val done')

balanced_acc_1_val=balanced_accuracy_score(labels_val, preds1_val, adjusted=True)
print('balanced_acc_1_ensemble_val',balanced_acc_1_val)
balanced_acc_2_val=balanced_accuracy_score(labels_val, preds2_val, adjusted=True)
print('balanced_acc_2_ensemble_val',balanced_acc_2_val)
balanced_acc_3_val=balanced_accuracy_score(labels_val, preds3_val, adjusted=True)
print('balanced_acc_3_val',balanced_acc_3_val)

balanced_acc_ensemble_val=balanced_accuracy_score(labels_val, preds_ensemble_val, adjusted=True)
print('balanced_acc_ensemble_val',balanced_acc_ensemble_val)

