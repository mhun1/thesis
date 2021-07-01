import torch
from torch import nn
import torch.nn.functional as F


class ReverseCrossEntropy(nn.Module):
    def __init__(self, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.scale = scale

    def forward(self, x, y):
        x = F.sigmoid(x)
        # x = torch.clamp(x, min=1e-7, max=1.0)
        y = torch.clamp(y, min=1e-7, max=1.0)
        rce = -1 * torch.sum(x * torch.log(y), dim=1)
        return self.scale * rce.mean()


class NormalizedReverseCrossEntropy(nn.Module):
    def __init__(self, scale=1.0):
        super(NormalizedReverseCrossEntropy, self).__init__()
        self.scale = scale

    def forward(self, x, y):
        x = F.softmax(x, dim=1)

        x = torch.clamp(x, min=1e-7, max=1.0)
        y = torch.clamp(y, min=1e-7, max=1.0)
        normalizor = 1 / 4
        rce = -1 * torch.sum(x * torch.log(y), dim=1)
        return self.scale * normalizor * rce.mean()


class NormalizedCrossEntropy(nn.Module):
    def __init__(self, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.scale = scale

    def forward(self, x, y):
        x = F.log_softmax(x, dim=1)
        nce = -1 * torch.sum(y * x, dim=0) / (-x.sum(dim=0))
        return self.scale * nce.mean()
