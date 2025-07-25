import torch
from torch import nn
from torchsummary import summary


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_conv_1=False):
        super().__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        if use_conv_1:
            self.conv3 = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0
            )
        else:
            self.conv3 = None
        
    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))

        if self.conv3 is not None:
            x = self.conv3(x)

        y = self.relu(x2 + x)
        return y


class ResNet(nn.Module):
    def __init__(self, Residual, in_channels=1, out_channels=10):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage1 = nn.Sequential(
            Residual(64, 64, 1, False),
            Residual(64, 64, 1, False)
        )

        self.stage2 = nn.Sequential(
            Residual(64, 128, 2, True),
            Residual(128, 128, 1, False)
        )

        self.stage3 = nn.Sequential(
            Residual(128, 256, 2, True),
            Residual(256, 256, 1, False)
        )

        self.stage4 = nn.Sequential(
            Residual(256, 512, 2, True),
            Residual(512, 512, 1, False)
        )

        self.block2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_features=1*512, out_features=out_channels)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.stage1(x1)
        x3 = self.stage2(x2)
        x4 = self.stage3(x3)
        x5 = self.stage4(x4)
        x6 = self.block2(x5)
        return x6


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    channel_size = 1
    height, width = 224, 224

    model = ResNet(Residual).to(device)
    print(summary(model, input_size=(channel_size, height, width)))