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

argument_list=str(sys.argv)
print('argument_list: [model,epoch,learning_rate,batch_size, y_balanced]',argument_list)
print('for example: [resnet18, 100, 6, 36, y]')
model=sys.argv[1]
epochs=int(sys.argv[2]) #integral positive
learning_rate=float('1e-'+str(sys.argv[3])) #float
batch_size=int(sys.argv[4]) #positive integral value
if sys.arg[5] =='y':
    balanced='_balanced'
else:
    balanced='_no_balanced'

data_dir = '/imatge/ltarres/work/data/ISIC-images/'
val_dataset = ImageFolderWithPaths(os.path.join(data_dir, 'val'), data_transforms['val'])
train_dataset = ImageFolderWithPaths(os.path.join(data_dir, 'train'), data_transforms['train'])

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model=torch.load('output_model_'+model+'_'+str(epochs)+'ep_lr'+str(sys.argv[3])+a_learning_rate+'_bs'+str(batch_size)+balanced+'.pth')
model.eval()

print()

#data_acc = torch.tensor([], dtype=torch.long, device='cuda:0')
#preds_acc = torch.tensor([], dtype=torch.long, device='cuda:0')
data_acc = torch.tensor([], dtype=torch.long, device='cuda:0')
preds_acc = torch.tensor([], dtype=torch.long, device='cuda:0')
for phase in ['train', 'val']:
    if phase == 'train':
        for inputs, labels, paths in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) #tensor([4,4....,4], device='cuda:0')
            #print('labels:',labels)
            #print('preds: ', preds)
            data_acc = torch.cat((data_acc, labels.data), 0)
            preds_acc = torch.cat((preds_acc, preds), 0)
        conf_matrix = confusion_matrix(data_acc.cpu(), preds_acc.cpu())
        plot_confusion_matrix(cm=conf_matrix,
                                  normalize=False,
                                  target_names=['actinic keratosis', 'basal cell \ncarcinoma', 'dermatofibroma',
                                                'melanoma', 'nevus', 'pigmented \nbenign keratosis', 'vascular lesion'],
                                  title='Confusion Matrix',
                                  save_file='/imatge/ltarres/PycharmProjects/test/loss_acc/confusion_matrix'+'_train'+'_model_'+model+'_'+str(epochs)+'ep_lr'+str(sys.argv[3])+'_bs'+str(batch_size)+balanced+'.jpg')
    else :
        for inputs, labels, paths in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # tensor([4,4....,4], device='cuda:0')
            # print('labels:',labels)
            # print('preds: ', preds)
            data_acc = torch.cat((data_acc, labels.data), 0)
            preds_acc = torch.cat((preds_acc, preds), 0)
        conf_matrix = confusion_matrix(data_acc.cpu(), preds_acc.cpu())
        plot_confusion_matrix(cm=conf_matrix,
                              normalize=False,
                              target_names=['actinic keratosis', 'basal cell \ncarcinoma', 'dermatofibroma',
                                            'melanoma', 'nevus', 'pigmented \nbenign keratosis', 'vascular lesion'],
                              title='Confusion Matrix',
                              save_file='/imatge/ltarres/PycharmProjects/test/loss_acc/confusion_matrix'+'_val'+'_model_'+model+'_'+str(epochs)+'ep_lr'+str(sys.argv[3])+'_bs'+str(batch_size)+balanced+ '.jpg')



