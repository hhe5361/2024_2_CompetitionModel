# importing all the libraries we need
from __future__ import print_function, division
import numpy as np
import random
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from model import EfficientNet
from utils import checkParams, train_model, evaluate
from optimTarget import learning_rate, training_epochs, schedule_steps

seed = 3334
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = EfficientNet(num_classes=10).cuda()
test_model = EfficientNet(num_classes=10).cuda()

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
