import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0 
        )
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2 
        )
        self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1 
        )
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1 
        )

        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1 
        )
        self.pooling5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.fc6 = nn.Linear(in_features=6*6*256, out_features=4096)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096)
        self.fc8 = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x1 = self.pooling1(x1)
        x2 = self.relu(self.conv2(x1))
        x2 = self.pooling2(x2)
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x5 = self.pooling5(x5)
        x6 = self.flatten(x5)
        x6 = self.relu(self.fc6(x6))
        x6 = F.dropout(x6, 0.5)
        x7 = self.relu(self.fc7(x6))
        x7 = F.dropout(x7, 0.5)
        x8 = self.fc8(x7)   # CrossEntropyLoss include softmax
        return x8


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = AlexNet().to(device)
    print(summary(model, input_size=(1,227,227)))