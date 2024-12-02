from utility import * 
from resnet import *
from aug import *
from googlenet import *

if __name__ == "__main__":
    model_name = 'googlenet'
    training_epochs = 50
    schedule_steps = 25
    learning_rate = 0.01
    batch_size = 64

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Model initialization (first time required download)
    if model_name == 'resnet18':
        model_ft = ResNet18(num_classes = 10).cuda()
    elif model_name == 'resnet34':
        model_ft = ResNet34(num_classes = 10).cuda()
    elif model_name == 'googlenet':
        model_ft = CompactGoogleNet(num_classes=10).cuda()

    # 파라미터 크기 확인
    pytorch_total_params = sum(p.numel() for p in model_ft.parameters())
    print(f"Number of parameters: {pytorch_total_params}")
    if int(pytorch_total_params) > 5000000:
        print('Your model has the number of parameters more than 5 millions..')
        sys.exit()

    model_ft = model_ft.to(device)

    torch.manual_seed(3334)

    # Optimization Loss function
    criterion = nn.CrossEntropyLoss()
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=learning_rate, weight_decay=1e-4)  # weight_decay 추가
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=schedule_steps, gamma=0.1) # Decay LR by a factor of 0.1 every <schedule_steps> epochs

    # train ㄱㄱ
    train_loader = DataLoader(combined_train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)

    model_ft, epc, trn, val = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, train_loader=train_loader, device=device, train_data=train_data, num_epochs=training_epochs)

    # test ㄱㄱ
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    predictions = evaluate(model_ft, criterion, test_loader, device, CompactGoogleNet)
