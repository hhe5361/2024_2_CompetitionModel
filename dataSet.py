import torchvision.transforms as transforms
import torchvision.datasets as datasets
import path
from hyperPrams import batch_size
from torch.utils.data import DataLoader
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
