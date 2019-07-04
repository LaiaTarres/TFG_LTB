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

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save_file=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    save_file:    Path of destination jpg, has to be a string

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name  # title of graph
                          save_file    = name_of_final_file   #name and path of the final file
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(save_file)


model_name=sys.argv[1]
print('el model_name que li has passat es:', model_name)

labels_train_file='model_output/train_labels_'+model_name+'.txt' #els labels son tots iguals independentment del model
labels_val_file='model_output/val_labels_'+model_name+'.txt'

preds_train_file='model_output/train_preds_'+model_name+'.txt'
preds_val_file='model_output/val_preds_'+model_name+'.txt'

with open(labels_train_file, 'r') as textFile: #ja esta be
  labels1t = textFile.readlines()
labels_train = [float(s.replace('\n', '')) for s in labels1t]
with open(labels_val_file, 'r') as textFile:
  labels1v = textFile.readlines()
labels_val = [float(s.replace('\n', '')) for s in labels1v]

with open(preds_train_file, 'r') as textFile: #ja està bé
  preds1t = textFile.readlines()
preds_train = [float(s.replace('\n', '')) for s in preds1t]
with open(preds_val_file, 'r') as textFile:
  preds1v = textFile.readlines()
preds_val = [float(s.replace('\n', '')) for s in preds1v]



for phase in ['train', 'val']:
    if phase == 'train':
        conf_matrix = confusion_matrix(labels_train, preds_train)
        plot_confusion_matrix(cm=conf_matrix,
                      normalize=False,
                      target_names=['actinic keratosis', 'basal cell \ncarcinoma', 'dermatofibroma',
                                    'melanoma', 'nevus', 'pigmented \nbenign keratosis', 'vascular lesion'],
                      title='Confusion Matrix',
                      save_file='/imatge/ltarres/PycharmProjects/test/loss_acc/confusion_matrix_train_' + model_name + '.jpg')
    else:
        conf_matrix = confusion_matrix(labels_val, preds_val)
        plot_confusion_matrix(cm=conf_matrix,
                              normalize=False,
                              target_names=['actinic keratosis', 'basal cell \ncarcinoma', 'dermatofibroma',
                                            'melanoma', 'nevus', 'pigmented \nbenign keratosis', 'vascular lesion'],
                              title='Confusion Matrix',
                              save_file='/imatge/ltarres/PycharmProjects/test/loss_acc/confusion_matrix' + '_val_' + model_name+ '.jpg')
