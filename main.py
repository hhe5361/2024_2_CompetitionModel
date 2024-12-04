# importing all the libraries we need
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from utils import checkParams, train_model, evaluate
from resnet import ResNet34
from dataset import combined_train_data, test_data

# Device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Hyperparameters
learning_rate = 0.01
training_epochs = 50
schedule_steps = 7
batch_size = 32
num_splits = 5  # K-Fold의 Fold 수

# Cross Validation Setup
kf = KFold(n_splits=num_splits, shuffle=True, random_state=3334)

# Cross Validation Training Loop
fold_results = []
for fold, (train_idx, val_idx) in enumerate(kf.split(combined_train_data)):
    print(f"Starting Fold {fold + 1}/{num_splits}")

    # Train/Validation 데이터 분리
    train_subset = Subset(combined_train_data, train_idx)
    val_subset = Subset(combined_train_data, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # 모델 초기화
    model_ft = ResNet34(num_classes=10).to(device)
    checkParams(model_ft)

    # Optimization Loss function
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=schedule_steps, gamma=0.1)

    # Training the model
    model_ft, epc, trn, val = train_model(
        model=model_ft,
        criterion=criterion,
        optimizer=optimizer_ft,
        scheduler=exp_lr_scheduler,
        device=device,
        dataloaders=dataloaders,  # train과 val DataLoader 전달
        num_epochs=training_epochs,
    )

    # Fold 결과 저장
    fold_results.append({
        'fold': fold + 1,
        'train_loss': trn[0],
        'train_acc': trn[1],
        'val_loss': val[0],
        'val_acc': val[1],
    })

# Test 데이터 평가
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
test_model = ResNet34(num_classes=10).to(device)
predictions = evaluate(test_model, device)

# Cross Validation 결과 요약
print("\nCross Validation Results:")
for result in fold_results:
    print(f"Fold {result['fold']}: Best Val Acc: {max(result['val_acc']):.4f}")

print("Done!")