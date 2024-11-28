# importing all the libraries we need
from __future__ import print_function, division
import numpy as np
import random
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from utils import checkParams, train_model, evaluate
from optimTarget import learning_rate, training_epochs, schedule_steps
from model.model import EfficientNet
from model.utils import get_model_params

seed = 3334
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 매개변수 가져오기
blocks_args, global_params = get_model_params('efficientnet-b0', override_params={'include_top': False})

model_ft = EfficientNet(blocks_args, global_params).cuda()
model_ft._fc = nn.Linear(global_params['width_coefficient'] * 1280, 10)  

test_model = EfficientNet(blocks_args, global_params).cuda()
test_model._fc = nn.Linear(global_params['width_coefficient'] * 1280, 10) 

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
predictions = evaluate(model=model_ft, test_model=test_model, criterion=criterion, device=device)
