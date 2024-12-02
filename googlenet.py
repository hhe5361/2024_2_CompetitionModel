from utility import *

class CompactInception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(CompactInception, self).__init__()
        # 1x1 Convolution
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # 1x1 -> 3x3 Convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3)
        )

        # 1x1 -> 5x5 Convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5)
        )

        # Max pooling -> 1x1 Convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # Concatenate along channel axis
        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs

class CompactGoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CompactGoogleNet, self).__init__()
        # Initial layers with reduced channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Compact Inception modules with reduced channels
        self.inception3a = CompactInception(64, 32, 48, 64, 8, 16, 16)  # Output: 128
        self.inception3b = CompactInception(128, 64, 64, 96, 16, 32, 32)  # Output: 224
        
        self.inception4a = CompactInception(224, 64, 64, 128, 16, 32, 32)  # Output: 256
        self.inception4b = CompactInception(256, 64, 64, 128, 16, 32, 32)  # Output: 256
        
        self.inception5a = CompactInception(256, 128, 64, 160, 16, 48, 48)  # Output: 384

        # Pooling and classification layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(384, num_classes)  # 맞춘 입력 차원

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.inception4a(x)
        x = self.inception4b(x)

        x = self.inception5a(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
