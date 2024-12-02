from utility import *
from torch.utils.data import ConcatDataset

# train, val aug
train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(degrees=15),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# 기존 train 폴더에서 데이터 불러오기
train_dir = "Galaxy10/train"
augmented_dir = "Galaxy10/augmented"

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
augmented_data = datasets.ImageFolder(augmented_dir, transform=train_transform)

# train 데이터와 augmented 데이터 병합
combined_train_data = ConcatDataset([train_data, augmented_data])

# 검증 및 테스트 데이터는 기존 방식 유지
val_dir = "Galaxy10/val"
val_data = datasets.ImageFolder(val_dir, transform=val_transform)

# test aug
test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_dir = "Galaxy10/test"
test_data = datasets.ImageFolder(test_dir, transform=test_transform)

print(f"Combined train data size: {len(combined_train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Test data size: {len(test_data)}")
