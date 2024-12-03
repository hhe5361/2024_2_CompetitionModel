# importing all the libraries we need
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from optimTarget import learning_rate, training_epochs, schedule_steps
from utils import checkParams, train_model, evaluate
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model_ft = models.squeezenet1_0(weights=None)
model_ft.classifier[-1] = nn.Conv2d(512, 10, kernel_size=1)
model_ft.num_classes = 10


test_model = models.squeezenet1_0(weights=None)
test_model.classifier[-1] = nn.Conv2d(512, 10, kernel_size=1)
test_model.num_classes = 10

model_ft= model_ft.cuda()
test_model = test_model.cuda()

checkParams(model_ft)

model_ft = model_ft.to(device)

# Optimization Loss function
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=schedule_steps, gamma=0.1)


# Training the model
model_ft, epc, trn, val = train_model(model = model_ft, criterion= criterion, optimizer= optimizer_ft, scheduler= exp_lr_scheduler, device= device , num_epochs=training_epochs)

# Evaluating the model
predictions = evaluate(test_model,device)
