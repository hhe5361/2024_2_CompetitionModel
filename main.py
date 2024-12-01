# importing all the libraries we need
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from optimTarget import learning_rate, training_epochs, schedule_steps
from utils import checkParams, train_model, evaluate
from torchvision.models import efficientnet_b0 as EfficientNet
# 모델 매개변수 가져오기
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = EfficientNet(weights = None)
#test_model = EfficientNet(weights = None)

# 마지막 분류기 계층에 접근
num_ftrs = model_ft.classifier[1].in_features

# 새로운 분류기 생성 및 교체
model_ft.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(num_ftrs, 10)
)
# test_model.classifier = nn.Sequential(
#     nn.Dropout(p=0.2, inplace=True),
#     nn.Linear(num_ftrs, 10)
#)

model_ft = model_ft.cuda()
# test_model = test_model.cuda()

checkParams(model_ft)
print(device)

model_ft = model_ft.to(device)

# Optimization Loss function
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=schedule_steps, gamma=0.1)


# Training the model
model_ft, epc, trn, val = train_model(model = model_ft, criterion= criterion, optimizer= optimizer_ft, scheduler= exp_lr_scheduler, device= device , num_epochs=training_epochs)

# Evaluating the model
predictions = evaluate(model=model_ft, criterion=criterion, device=device)
