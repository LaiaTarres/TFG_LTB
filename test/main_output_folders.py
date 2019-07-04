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
import glob
import shutil

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
test_dataloader = DataLoader(val_dataset, batch_size=36, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model=torch.load('output_model_3.pth')
model.eval()

idx_to_class = {val: key for key, val in val_dataset.class_to_idx.items()}

for inputs, labels, paths in test_dataloader: #tqdm(test_dataloader, total=len(test_dataloader)):
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1) #tensor([4,4....,4], device='cuda:0')
    for i in range(0,len(labels)): #volem que vagi de 0 a 36
        src_dir = str(paths[i]) #per exemple /imatge/ltarres/work/data/ISIC-images/val/nevus/ISIC_0026183.jpg
        dst_dir_aux=src_dir.split('/val/') #separar-lo
        dst_dir_aux_2=dst_dir_aux[1].split('/')
        dst_dir = dst_dir_aux[0] +'/output/' + str(idx_to_class.get(preds[i].item())) #+ '/' + dst_dir_aux_2[1] #si volem mantenir el nom, no cal
        print('src_dir',src_dir)
        print('dst_dir',dst_dir)
        shutil.copy(src_dir, dst_dir)