import torch
from torch import nn
from torchsummary import summary


class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, dropout_rate=0.0):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0
        )

        self.bn2 = nn.BatchNorm2d(num_features=bn_size * growth_rate)
        self.conv2 = nn.Conv2d(
            in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x1 = self.conv1(self.relu(self.bn1(x)))
        x2 = self.conv2(self.relu(self.bn2(x1)))

        if self.dropout is not None:
            x2 = self.dropout(x2)

        y = torch.cat([x, x2], dim=1)
        return y


class _DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, bn_size, dropout_rate=0.0):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(
                in_channels=in_channels + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout_rate=dropout_rate
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU()

        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.conv(self.relu(self.bn(x)))
        x2 = self.pooling(x1)
        return x2


class DenseNet(nn.Module):
    def __init__(self, in_channels=1, init_channels=64, out_channels=10, growth_rate=32, bn_size=4, dropout_rate=0.0, block_config=(6, 12, 24, 16)):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=init_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=init_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_channels = init_channels

        self.Dense1 = _DenseBlock(
            in_channels=num_channels, num_layers=block_config[0], growth_rate=growth_rate, bn_size=bn_size, dropout_rate=dropout_rate
        )
        num_channels += block_config[0] * growth_rate
        self.Trans1 = _Transition(in_channels=num_channels, out_channels=num_channels // 2)
        num_channels //= 2

        self.Dense2 = _DenseBlock(
            in_channels=num_channels, num_layers=block_config[1], growth_rate=growth_rate, bn_size=bn_size, dropout_rate=dropout_rate
        )
        num_channels += block_config[1] * growth_rate
        self.Trans2 = _Transition(in_channels=num_channels, out_channels=num_channels // 2)
        num_channels //= 2

        self.Dense3 = _DenseBlock(
            in_channels=num_channels, num_layers=block_config[2], growth_rate=growth_rate, bn_size=bn_size, dropout_rate=dropout_rate
        )
        num_channels += block_config[2] * growth_rate
        self.Trans3 = _Transition(in_channels=num_channels, out_channels=num_channels // 2)
        num_channels //= 2

        self.Dense4 = _DenseBlock(
            in_channels=num_channels, num_layers=block_config[3], growth_rate=growth_rate, bn_size=bn_size, dropout_rate=dropout_rate
        )
        num_channels += block_config[3] * growth_rate
        
        self.block2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=num_channels, out_features=out_channels)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.Dense1(x1)
        x2 = self.Trans1(x2)
        x3 = self.Dense2(x2)
        x3 = self.Trans2(x3)
        x4 = self.Dense3(x3)
        x4 = self.Trans3(x4)
        x5 = self.Dense4(x4)
        x6 = self.block2(x5)
        return x6


class DenseNet_121(DenseNet):
    def __init__(self, in_channels=1, out_channels=10, growth_rate=32, bn_size=4, dropout_rate=0.0):
        super().__init__(in_channels=in_channels, init_channels=64, out_channels=out_channels,
                         growth_rate=growth_rate, bn_size=bn_size, dropout_rate=dropout_rate,
                         block_config=(6, 12, 24, 16))
        
    def forward(self, x):
        return super().forward(x)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    channel_size = 1
    height, width = 224, 224

    model = DenseNet_121(channel_size, 10, 32, 4, 0.0).to(device)
    print(summary(model, input_size=(channel_size, height, width)))