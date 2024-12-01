import sys
import numpy as np
import time
import torch
import copy
import itertools
from dataaug import train_loader, train_data, test_loader
import pandas as pd

def checkParams(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")
    if int(pytorch_total_params) > 5000000:
        print('Your model has the number of parameters more than 5 millions..')
        sys.exit()

def train_model(model, criterion, optimizer, scheduler, device,  num_epochs=25):
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

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data batches
            for inputs, labels in train_loader:
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
              trn_acc_lst.append(np.round(epoch_acc.cpu().item(),4 ))

            elif phase == 'val':
              val_loss_lst.append(np.round(epoch_loss, 4))
              val_acc_lst.append(np.round(epoch_acc.cpu().item(), 4))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'model_best.pt')
                best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(),'model_latest.pt')

    trn_metadata = [trn_loss_lst, trn_acc_lst]
    val_metadata = [val_loss_lst, val_acc_lst]

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_lst, trn_metadata, val_metadata

def evaluate(model, device, criterion):
    count = 0
    all_count = 0
    model.eval()  # setting the model to evaluate mode
    preds = []
    gts = []


    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)  # labels도 GPU로 이동

        # predicting
        with torch.no_grad():
            outputs = model(inputs)
            _, pred = torch.max(outputs, dim=1)  # 예측값
            preds.append(pred.cpu())  # GPU -> CPU 이동 후 저장
            gts.append(labels.cpu())  # 정답값도 저장

            # 정확도 계산
            count += (pred == labels).sum().item()
        all_count += labels.size(0)  # 현재 배치의 샘플 수 추가

    # 최종 정확도 출력
    accuracy = count / all_count
    print(f'Accuracy: {accuracy:.4f}')

    # 예측값 저장
    category = [t.numpy() for t in preds]
    t_category = list(itertools.chain(*category))
    Id = list(range(0, len(t_category)))

    prediction = {
        'Id': Id,
        'Category': t_category
    }

    prediction_df = pd.DataFrame(prediction, columns=['Id', 'Category'])

    # 저장경로는 변경하셔도 됩니다.
    prediction_df.to_csv('prediction.csv', index=False)

    print('Done!!')
    return preds

