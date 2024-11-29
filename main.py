# importing all the libraries we need
import torch
from torchvision.models import densenet121
from torch import nn, optim
from torch.optim import lr_scheduler
from optimTarget import learning_rate, training_epochs, schedule_steps
from utils import checkParams, train_model, evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = densenet121(pretrained=False).cuda()
test_model = densenet121(pretrained=False).cuda()

model_ft.classifier = nn.Linear(model_ft.classifier.in_features, 10)
test_model.classifier = nn.Linear(model_ft.classifier.in_features, 10)

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
