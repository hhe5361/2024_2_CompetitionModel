import torch
import torchsummary
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from util import train_model, checkParams, evaluate
from model import ResNet18
from hyperPrams import training_epochs, schedule_steps, learning_rate, batch_size

def main():
    seed = 3334
    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #insert your model here

    model_ft = ResNet18(num_classes = 10).cuda()
    checkParams(model_ft)
    torchsummary.summary(model_ft, (3,256,256))
    
    # Optimization Loss function
    criterion = nn.CrossEntropyLoss()

    # Parameters and optimization rate
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)

    # Decay LR by a factor of 0.1 every <schedule_steps> epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=schedule_steps, gamma=0.1)

    train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, device, num_epochs=training_epochs)

    # testing the model
    predictions = evaluate(model_ft, ResNet18(num_classes=10), criterion, device)

if __name__ == '__main__':
    main()
