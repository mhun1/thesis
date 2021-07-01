import torch
import torch.nn as nn


def weight_init(module):
    if isinstance(module, nn.Linear) or isinstance(module,nn.Conv3d):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            module.bias.data.zero_()


class MetaLoss(nn.Module):
    def __init__(self, in_=200704, hid_=128):
        super(MetaLoss, self).__init__()

        self.activation = nn.ReLU()
        self.loss = nn.Sequential(
            nn.Linear(in_, 64, bias=False),
            self.activation,
            nn.Linear(64, 32, bias=False),
            self.activation,
            nn.Linear(32, 1, bias=False),
            nn.Softplus(),
        )
        self.loss.apply(weight_init)

    def forward(self, pred,y):
        x = torch.cat((pred,y), dim=1)
        x = x.flatten()

        return self.loss(x).mean()

class MetaLoss3D(nn.Module):
    def __init__(self, in_=2, hid_=128):
        super(MetaLoss3D, self).__init__()

        self.activation = nn.ReLU()
        self.loss = nn.Sequential(
            nn.Conv3d(in_, 4, 1, bias=False),
            self.activation,
            nn.Conv3d(4, 3, 1, bias=False),
            self.activation,
            nn.Conv3d(3, 2, 1,bias=False),
            self.activation,
            nn.Conv3d(2, 1, 1, bias=False),
            nn.Softplus(),
        )
        self.loss.apply(weight_init)

    def forward(self, pred, y):
        x = torch.cat((pred,y), dim=1)
        return self.loss(x).mean()

class MetaLoss2D(nn.Module):
    def __init__(self, in_=2, hid_=128):
        super(MetaLoss2D, self).__init__()

        self.activation = nn.ReLU()

        self.loss = nn.Sequential(
            nn.Conv2d(in_, 16, 1, bias=False),
            self.activation,
            nn.Conv2d(16, 8, 1, bias=False),
            self.activation,
            nn.Conv2d(8, 4, 1,bias=False),
            self.activation,
            nn.Conv2d(4, 1, 1, bias=False),
            nn.Softplus(),
        )
        self.loss.apply(weight_init)

    def forward(self, pred, y):
        x = torch.cat((pred,y), dim=1)
        return self.loss(x).mean()
