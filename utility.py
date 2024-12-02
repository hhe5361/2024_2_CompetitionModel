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
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.optim import lr_scheduler
import copy
from random import shuffle
import torch.nn.functional as F
import itertools
from torch.autograd import Function
import pywt
from einops import rearrange
from einops.layers.torch import Rearrange

def train_model(model, criterion, optimizer, scheduler, train_loader, device, train_data, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_lst = []
    trn_loss_lst = []
    trn_acc_lst = []
    val_loss_lst = []
    val_acc_lst = []

    for epoch in range(num_epochs):
        print('-' * 50)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval() 

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data batches
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
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
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_acc = running_corrects.double() / len(train_data)
            epoch_loss = running_loss / len(train_data)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # save training val metadata metrics
            if phase == 'train':
              epoch_lst.append(epoch)
              trn_loss_lst.append(np.round(epoch_loss,4 ))
              trn_acc_lst.append(np.round(epoch_acc.cpu().item(), 4))

            elif phase == 'val':
            	val_loss_lst.append(np.round(epoch_loss, 4))
            	val_acc_lst.append(np.round(epoch_acc.cpu().item(), 4))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'googlebest.pt')
                best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(),'googlelatest.pt')

    trn_metadata = [trn_loss_lst, trn_acc_lst]
    val_metadata = [val_loss_lst, val_acc_lst]

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_lst, trn_metadata, val_metadata


# testing how good the model is
def evaluate(criterion, test_loader, device, CompactGoogleNet):
    test_model = CompactGoogleNet(num_classes=10).cuda()
    test_model.eval()       # setting the model to evaluate mode
    preds = []
    gts = []
    Category = []

    #저장경로는 변경하셔도 됩니다.
    test_model.load_state_dict(torch.load('googlebest.pt'))

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        # predicting
        with torch.no_grad():

            outputs = test_model(inputs)
            _,pred = torch.max(outputs,dim=1)
            preds.append(pred)

    category = [t.cpu().numpy() for t in preds]

    t_category = list(itertools.chain(*category))

    Id = list(range(0, len(t_category)))

    prediction = {
      'Id': Id,
      'Category': t_category
    }

    prediction_df = pd.DataFrame(prediction, columns=['Id','Category'])

    #저장경로는 변경하셔도 됩니다.
    prediction_df.to_csv('googlenet.csv', index=False)

    print('Done!!')

    return preds
