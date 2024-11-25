import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=3, out_channels=num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=num_features, out_channels=3, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.dncnn(x)
        return residual - out

def get_model(num_layers=17, num_features=64):
    return DnCNN(num_layers, num_features)