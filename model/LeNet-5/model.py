import torch
from torch import nn
from torchsummary import summary


class LeNet_5(nn.Module):
    def __init__(self, in_channels=1, out_channels=10):
        super().__init__()
        self.sig = nn.Sigmoid()
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2
        )
        self.subsampling1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        )
        self.subsampling2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc3 = nn.Linear(in_features=5*5*16, out_features=120)
        self.fc4 = nn.Linear(in_features=120, out_features=84)
        self.fc5 = nn.Linear(in_features=84, out_features=out_channels)
    
    def forward(self, x):
        x1 = self.sig(self.conv1(x))
        x1 = self.subsampling1(x1)
        x2 = self.sig(self.conv2(x1))
        x2 = self.subsampling2(x2)
        x3 = self.flatten(x2)
        x3 = self.sig(self.fc3(x3))
        x4 = self.sig(self.fc4(x3))
        x5 = self.fc5(x4)   # CrossEntropyLoss include softmax
        return x5

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    channel_size = 1
    height, width = 28, 28

    model = LeNet_5().to(device)    # mark
    # input shape: [batch_size, channel_size, height, width]
    print(summary(model, input_size=(channel_size, height, width)))