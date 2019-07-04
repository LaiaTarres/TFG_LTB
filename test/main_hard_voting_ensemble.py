import torch
import sys
from sklearn.metrics import balanced_accuracy_score

def hardVotingClassifier(preds_model1, preds_model2, preds_model3):
    """Se li passen els 4 tensors amb la info acumulada. S'ha de fer la moda de cada imatge"""
    print('type(preds_model1)',type(preds_model1))
    hard_voting_preds=[]
    for i in range(len(preds_model1)): #range perquè li estavem tensor amb un vector de mida batch size, aquí ja li passarem un vector
        list_preds = [preds_model1[i], preds_model2[i], preds_model3[i]]#llista amb les tres posicions
        hard_voting_preds.append(max(set(list_preds), key=list_preds.count))  # Returns the highest occurring item

    return hard_voting_preds#torch.tensor(hard_voting_preds, dtype=torch.long, device=device)

model1=sys.argv[1]
model2=sys.argv[2]
model3=sys.argv[3]
#print('se li ha de passar el model sense pth [output_model_my_resnet_18_2018, ]', model)
print('srun --mem 16G python main_hard_voting_ensemble.py output_model_my_resnet_18_2018 output_model_my_resnet_34_2018 output_model_my_resnet_50_2018')

labels_train_file='model_output/train_labels_'+model1+'.txt' #els labels son tots iguals independentment del model
labels_val_file='model_output/val_labels_'+model1+'.txt'

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

preds_ensemble_train=hardVotingClassifier(preds1_train,preds2_train,preds3_train)
preds_ensemble_val=hardVotingClassifier(preds1_val, preds2_val, preds3_val)

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