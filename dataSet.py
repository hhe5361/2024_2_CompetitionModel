import torchvision.transforms as transforms
import torchvision.datasets as datasets
import path
from hyperPrams import batch_size
from torch.utils.data import DataLoader
import torch
seed = 3334
torch.manual_seed(seed)

#data augmentation
train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_data = datasets.ImageFolder(path.train_dir, train_transform)
val_data = datasets.ImageFolder(path.val_dir, val_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

train_data.transform = train_transform
val_data.transform = val_transform

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

#test data
test_transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_data = datasets.ImageFolder(path.test_dir)
test_data.transform = test_transform
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
