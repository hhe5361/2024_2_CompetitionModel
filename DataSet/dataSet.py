# importing all the libraries we need
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import random
import pandas as pd
import torch
from torch import nn, cuda, optim
from torchvision import models,transforms,datasets
from torch.utils.data import DataLoader,random_split
from PIL import Image
import seaborn as sns
import h5py
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.optim import lr_scheduler
import copy
from random import shuffle
import torch.nn.functional as F

seed = 3334
torch.manual_seed(seed)

#directory 주소 변경 알아서 잘 해야 함. 
with h5py.File('Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images']) # Select only few examples [0:5000]
    labels = np.array(F['ans'])

print('Data loaded successfully')

class_names = ['Disturbed', 'Merging', 'Round_Smooth',
               'In-between_Round_Smooth', 'Cigar_Shaped_Smooth', 'Barred_Spiral',
               'Unbarred_Tight_Spiral', 'Unbarred_Loose_Spiral', 'Edge-on_without_Bulge',
               'Edge-on_with_Bulge']

labels_cat = torch.nn.functional.one_hot(torch.tensor(labels), 10).numpy()
print('Categorical label:', labels_cat[0])
print('Shape of data structure labels {} and images {}'.format(labels_cat.shape, images.shape))
print('Dataset images per class:', np.sum(labels_cat, axis=0))

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state = 3444)
for train_idx, valt_idx in sss.split(labels, labels):
    train_labels = labels_cat[train_idx]
    valt_labels = labels_cat[valt_idx]
    print("TRAIN indexes:", train_idx, "number of objects:", train_idx.shape)
    print(np.sum(train_labels, axis=0))
    print("Valtest indexes:", valt_idx, "number of objects:", valt_idx.shape)
    print(np.sum(valt_labels, axis=0))

train_labels = labels[train_idx]
valt_labels = labels[valt_idx]
train_images = images[train_idx]
valt_images = images[valt_idx]

valtest_labels_cat = torch.nn.functional.one_hot(torch.tensor(valt_labels), 10).numpy()
print('Categorical label:', valtest_labels_cat[0])
print('Shape of data structure labels {} and images {}'.format(valtest_labels_cat.shape, valt_images.shape))
print('Dataset images per class:', np.sum(valtest_labels_cat, axis=0))

torch.manual_seed(3334)

ssss = StratifiedShuffleSplit(n_splits=1, test_size = 0.5, random_state = 3444)
for val_idx, test_idx in ssss.split(valt_labels, valt_labels):
    val_labels = valtest_labels_cat[val_idx]
    test_labels = valtest_labels_cat[test_idx]
    print("Validation indexes:", val_idx, "number of objects:", val_idx.shape)
    print(np.sum(val_labels, axis=0))
    print("TEST indexes:", test_idx, "number of objects:", test_idx.shape)
    print(np.sum(test_labels, axis=0))

val_labels = valt_labels[val_idx]
test_labels = valt_labels[test_idx]
val_images = valt_images[val_idx]
test_images = valt_images[test_idx]

main = os.getcwd() #현재 작업 디렉토리를 main 변수에 저장. 디렉 구조 생성한 뒤 다시 이 위치로 돌아오기 위해 사용?

# 1.Create directory tree and save images
dataset_dirname = 'Galaxy10' # 최상위 디렉 만들기

try:
    os.mkdir(dataset_dirname)
except OSError:
    print('OSError: Creating or already exists the directory')
# Main dataset folder
os.chdir(dataset_dirname)

# 2.Train partition
os.mkdir('train') #train folder 생성
os.chdir('train') #이동
for cls in class_names:   # for each class 'cls' 각 클래스 이름 별로 하위 디렉 생성
    os.mkdir(cls)
    # train/<class> folder save images
    cls_int = class_names.index(cls)
    print('Train - Class: ', cls)
    for i in range(len(train_labels)):   # traverse all train lavels
        if train_labels[i] == cls_int:   # save instance 'i' belong to class 'cls'
            img_path = os.path.join(os.getcwd(), cls)
            #print('Save image {} in {}'.format(i, img_path))
            img = Image.fromarray(train_images[i])
            img.save(os.path.join(img_path, 'galaxy10_img{}.jpg'.format(i)))

# 3.Validation partition
os.chdir(os.path.join(main, dataset_dirname))
os.mkdir('val')
os.chdir('val')
for cls in class_names:
    os.mkdir(cls)
    # val/<class> folder save images
    cls_int = class_names.index(cls)
    print('Validation - Class: ', cls)
    for i in range(len(val_labels)):   # traverse all train lavels
        if val_labels[i] == cls_int:   # save instance 'i' belong to class 'cls'
            img_path = os.path.join(os.getcwd(), cls)
            #print('Save image {} in {}'.format(i, img_path))
            img = Image.fromarray(val_images[i])
            img.save(os.path.join(img_path, 'galaxy10_img{}.jpg'.format(i)))

# 3.test partition
os.chdir(os.path.join(main, dataset_dirname))
os.mkdir('test')
os.chdir('test')
for cls in class_names:
    os.mkdir(cls)
    # val/<class> folder save images
    cls_int = class_names.index(cls)
    print('Test - Class: ', cls)
    for i in range(len(test_labels)):   # traverse all train lavels
        if test_labels[i] == cls_int:   # save instance 'i' belong to class 'cls'
            img_path = os.path.join(os.getcwd(), cls)
            #print('Save image {} in {}'.format(i, img_path))
            img = Image.fromarray(test_images[i])
            img.save(os.path.join(img_path, 'galaxy10_img{}.jpg'.format(i)))

os.chdir(main)
