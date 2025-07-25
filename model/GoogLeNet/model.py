import torch
from torch import nn
from torchsummary import summary


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        self.relu = nn.ReLU()

        self.path1_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=c1, kernel_size=1, stride=1, padding=0
        )

        self.path2_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=c2[0], kernel_size=1, stride=1, padding=0
        )
        self.path2_2 = nn.Conv2d(
            in_channels=c2[0], out_channels=c2[1], kernel_size=3, stride=1, padding=1
        )

        self.path3_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=c3[0], kernel_size=1, stride=1, padding=0
        )
        self.path3_2 = nn.Conv2d(
            in_channels=c3[0], out_channels=c3[1], kernel_size=5, stride=1, padding=2
        )

        self.path4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.path4_2 = nn.Conv2d(
            in_channels=in_channels, out_channels=c4, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        p1 = self.relu(self.path1_1(x))
        p2 = self.relu(self.path2_2(self.relu(self.path2_1(x))))
        p3 = self.relu(self.path3_2(self.relu(self.path3_1(x))))
        p4 = self.relu(self.path4_2(self.path4_1(x)))

        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, Inception, in_channels=1, out_channels=10):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.block4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (128, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_features=1*1024, out_features=out_channels)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            else:
                pass

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        return x5


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    channel_size = 1
    height, width = 224, 224

    model = GoogLeNet(Inception).to(device)
    print(summary(model, input_size=(channel_size, height, width)))