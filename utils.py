import sys
import numpy as np
import time
import torch
import copy
import itertools
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import pandas as pd

def checkParams(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")
    if int(pytorch_total_params) > 5000000:
        print('Your model has the number of parameters more than 5 millions..')
        sys.exit()

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25):
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
        print(f'Epoch {epoch + 1}/{num_epochs}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                epoch_lst.append(epoch)
                trn_loss_lst.append(round(epoch_loss, 4))
                trn_acc_lst.append(round(epoch_acc.cpu().item(), 4))
                scheduler.step()
            else:
                val_loss_lst.append(round(epoch_loss, 4))
                val_acc_lst.append(round(epoch_acc.cpu().item(), 4))

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'model_best.pt')
                    best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), 'model_latest.pt')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, epoch_lst, [trn_loss_lst, trn_acc_lst], [val_loss_lst, val_acc_lst]

def evaluate(test_model, test_loader, device):
    count = 0
    all_count = 0
    preds = []

    test_model.load_state_dict(torch.load('model_best.pt'))
    test_model.to(device)
    test_model.eval()

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = test_model(inputs)
            _, pred = torch.max(outputs, dim=1)
            preds.append(pred.cpu())

            count += (pred == labels).sum().item()
        all_count += labels.size(0)

    accuracy = count / all_count
    print(f'Accuracy: {accuracy:.4f}')

    category = [t.numpy() for t in preds]
    t_category = list(itertools.chain(*category))
    Id = list(range(0, len(t_category)))

    prediction = {
        'Id': Id,
        'Category': t_category
    }

    prediction_df = pd.DataFrame(prediction, columns=['Id', 'Category'])
    prediction_df.to_csv('prediction.csv', index=False)

    print('Done!!')
    return preds