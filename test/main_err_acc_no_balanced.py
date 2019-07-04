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
import sys

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses #vector amb tants zeros com nombre de classes
    for item in images: #conta les imatges que hi ha a cada classe
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses #vector amb tants 0. com nombre de classes
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i]) #total/el nombre d'imatges per aquella classe
    weight = [0] * len(images) #defineix weight com un vector de zeros de longitud nombre d'imatges
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]] #aqui fa, per cada imatge, li assigna el weight de la classe
    return weight

def make_weights_per_class(images, nclasses):
    count = [0] * nclasses  # vector amb tants zeros com nombre de classes
    for item in images:  # conta les imatges que hi ha a cada classe
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses  # vector amb tants 0. com nombre de classes
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])  # total/el nombre d'imatges per aquella classe
    return weight_per_class

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomRotation(180), #multiples de 90, 90, 180, 270
        transforms.RandomRotation(270),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
} #per normalitzar les imatges.

argument_list=str(sys.argv)
print('argument_list: [model,epoch,learning_rate,batch_size]',argument_list)
print('for example: [resnet18, 100, 6, 36]')
model=sys.argv[1]
epochs=int(sys.argv[2]) #integral positive
learning_rate=float('1e-'+str(sys.argv[3])) #float
batch_size=int(sys.argv[4]) #positive integral value

data_dir = '/imatge/ltarres/work/data/ISIC-images/' #tinc moltes carpetes, dues d'elles son train i val.

escriptura_loss_val = '/imatge/ltarres/PycharmProjects/test/loss_acc/val_loss_'+model+'_'+str(epochs)+'ep_lr'+str(sys.argv[3])+'_bs'+str(batch_size)+'_no_balanced.txt'
escriptura_loss_train = '/imatge/ltarres/PycharmProjects/test/loss_acc/train_loss_'+model+'_'+str(epochs)+'ep_lr'+str(sys.argv[3])+'_bs'+str(batch_size)+'_no_balanced.txt'
escriptura_acc_train = '/imatge/ltarres/PycharmProjects/test/loss_acc/train_acc_'+model+'_'+str(epochs)+'ep_lr'+str(sys.argv[3])+'_bs'+str(batch_size)+'_no_balanced.txt'
escriptura_acc_val = '/imatge/ltarres/PycharmProjects/test/loss_acc/val_acc_'+model+'_'+str(epochs)+'ep_lr'+str(sys.argv[3])+'_bs'+str(batch_size)+'_no_balanced.txt'
open(escriptura_loss_val, 'w').close() #borrem els continguts abans de res
open(escriptura_loss_train, 'w').close() #borrem els continguts abans de res
open(escriptura_acc_train, 'w').close() #borrem els continguts abans de res
open(escriptura_acc_val, 'w').close() #borrem els continguts abans de res


train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']) #mirar si ho fa random
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

print('Classes to index', train_dataset.class_to_idx)

weights_train = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes)) #crees els weights per les images i les classes
weights_train = torch.DoubleTensor(weights_train) #n-dimensional array, és més fàcil per operar que una matriu
sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights_train, len(weights_train)) #agafa mostres amb la probabilitat de weights_train

weights_val = make_weights_for_balanced_classes(val_dataset.imgs, len(val_dataset.classes)) #igual pero per val
weights_val = torch.DoubleTensor(weights_val)
sampler_val = torch.utils.data.sampler.WeightedRandomSampler(weights_val, len(weights_val))


#classes no balancejades:
#train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
#val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=4, shuffle=True)

#classes balancejades:
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=sampler_train) #combina un dataset i un sampler. Agafa dades del train set, la mida del batch, el nombre de subprocessos per carregar les dades i el sampler (quines dades s'han d'agafar)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, sampler=sampler_val)


dataloaders = {'train': train_dataloader, 'val': val_dataloader}

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = train_dataset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=5)
#f = open( 'fitxer_loss_acc_100epochs.txt', 'a+' ) #fitxer de sortida de loss i accuracy

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1000.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data) #casos que les prediccions son iguals que la data

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                with open(escriptura_loss_train, 'a+') as f:
                    f.write('{}\n'.format(epoch_loss))
                with open(escriptura_acc_train, 'a+') as f:
                    f.write('{}\n'.format(epoch_acc))
            if phase == 'val':
                # copy loss,acc val in a file
                with open(escriptura_loss_val, 'a+') as f:
                    f.write('{}\n'.format(epoch_loss))
                with open(escriptura_acc_val, 'a+') as f:
                    f.write('{}\n'.format(epoch_acc))

            # deep copy the model
            # cambiar per loss de validacio (mínima, no maxima)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc=epoch_acc
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    #f.close()#tancar el fitxer
    return model

if model=='resnet18':
    model_ft = models.resnet18(pretrained=True) #carregar un model amb uns weights que ja han fet un train, es descarreguen els weights. #s'entrena el model amb la resnet18,
elif model=='resnet34':
    model_ft = models.resnet34(pretrained=True)
elif model=='resnet50':
    model_ft = models.resnet50(pretrained=True)
elif model=='resnet101':
    model_ft = models.resnet101(pretrained=True)
elif model=='resnet152':
    model_ft = models.resnet152(pretrained=True)
else :
    model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features #numero de features
model_ft.fc = nn.Linear(num_ftrs, 7) #aplica transformacio linear y=Atx+b, nou model de 2 classes, nosaltres ho haurem de canviar a 7 (l'imagen net, per defecte son 1000)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss() #combina nn.LogSoftmax() i nn.NLLLoss() en una sola classe. És útil quan tenim un nombre determinat de classes. És la manera de calcular la loss
#mirar que aquest criteri utilitzi els weights. crec que li pots passar de paràmeter els weights, però ha de ser un tensor
#https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
#el problema és que tenim els weights a dins de la funció... i per tant s'hauria de reassignar depenent de si és val o és train... per tant ho fem a dalt

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate) #s'optimitza amb el mètode d'Adam, amb aquest learning rate que normalment és el que s'ha de provar per veure quin funciona

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0)
exp_lr_scheduler = None

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=epochs) #s'entrena el model amb la resnet18,

torch.save(model_ft, 'output_model_'+model+'_'+str(epochs)+'ep_lr'+str(sys.argv[3])+'_bs'+str(batch_size)+'_no_balanced.pth') #canviar això per saber quin és


#a l'entrar: source VirtualEnvs/venv/bin/activate
# module load python/3.6.2 cuda/8.0
#cd PycharmProjects/test/
# per correr srun --mem 16GB --gres=gpu:1 python main_err_acc_no_balanced.py #el sergio diu de tambe afegir-li --mem 16GB i provar si així funciona...

#descarregar la base de dades amb metadades, i penjar-les al servidor
#dividir la base de dades en les classes (fet, dins de HAM1000)
#dividim en train i test, la idea es que es conservi el balanç de classes stratified train test plead?
#7 clases balanceadas?

#entrenem la resnet i anem provant fins que hi hagi alguna cosa que funcioni més o menys bé (comparant amb les que ja hi ha)

#després ja ve metadades i confiança
#paper de no pigmentado i paper de metadatos

#anar jugant, treure el weighted random sample, perque iguali les classes i no doni preferència i li sigui fàcil (que digui sempre el mateix)
#una altra opció seria donarli més pes a la classe no representada a la loss, però no va be perque continues entrenant mes mab una classe que amb l'atre. Va lent i quan troba un del que hi ha poc fa un pic


#hem fet un nou tmux perque descarregui la base de dades directament al servidor
#per fer un tmux: tmux new -s hola #hola es el nom que volem de la sessio
#tmux a -t hola #per entrar a la sessio hola si has sortit
#per sortir de la sessio no fer exit, sino es peta, sino control + b + d (o algo així, sino buscar online)

#per correr aquest script hauria de fer srun

#25/04/2019, apunts:
#primer de tot hem posat prints per veure el que estava passant.
#Els prints de l'input el que mostren és el vector d'inputs [16 imatges, 3 canals, 224, 224] --perquè hem fet resize al principi pq les imatges fossin quadrades i després el vector de labels de mida 16, que son les etiquetes correctes de cada imatge
#Els prints de l'output, el que mostres és un vector [16,7] amb les probabilitats que té cada imatge de ser de cada classe, i es queda amb el màxim de cada fila com les prediccions
#Després es fa l'error entre l'input i la predicció, i aquest error és el que es vol optimitzar

#Ara el que hem de fer son les corbes de loss per train i per val. Les de train tendeixen a decreixer pero a val pot ser que comencin a pujar perquè hi haurà overfitting. Has de trobar el punt on lo de val sigui mínim.

#Tot el que s'obre per terminal es pot fer amb nano

#nombre de coses dins d'una carpeta: ls | wc -l

#per copiar cp source destination

#per eliminar un directori i tot el que hi ha dins: rm -rf directory