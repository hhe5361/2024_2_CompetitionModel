from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch
import numpy as np
import random

# Seed 설정
seed = 3334
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# 데이터 변환 정의
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 수평 뒤집기
    transforms.RandomVerticalFlip(p=0.5),    # 50% 확률로 수직 뒤집기
    transforms.RandomRotation(degrees=15),   # ±15도 내에서 랜덤 회전
    transforms.ToTensor(),                   # 텐서로 변환
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 정규화
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Train 데이터 경로들
train_dir_1 = "/data/coqls1229/repos/mip_res34/Galaxy10/train"
train_dir_2 = "/data/a7381pp/repos/mlip/Galaxy10/aug"

# 데이터셋 로드
train_data_1 = datasets.ImageFolder(train_dir_1, transform=train_transform)
train_data_2 = datasets.ImageFolder(train_dir_2, transform=train_transform)

# 두 데이터셋을 합침
combined_train_data = ConcatDataset([train_data_1, train_data_2])

# Validation 및 Test 데이터 로드
val_dir = "/data/coqls1229/repos/mip_res34/Galaxy10/val"  # 실제 경로로 변경 필요
test_dir = "/data/coqls1229/repos/mip_res34/Galaxy10/test"       # 실제 경로로 변경 필요
val_data = datasets.ImageFolder(val_dir, transform=val_transform)
test_data = datasets.ImageFolder(test_dir, transform=test_transform)

# DataLoader 생성
batch_size = 32  # 필요한 배치 크기로 변경
train_loader = DataLoader(combined_train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)