import torch
from torch import nn


class MCC(nn.Module):
    def __init__(self, epsilon=1, apply_non_lin=True):
        super(MCC, self).__init__()

        self.apply_non_lin = apply_non_lin
        self.epsilon = epsilon

    def forward(self, x, y):
        if self.apply_non_lin:
            x = x.sigmoid()

        tp = torch.sum(x * y)
        fp = torch.sum(x * (1 - y))
        fn = torch.sum((1 - x) * y)
        tn = torch.sum((1 - x) * (1 - y))

        num = (tp * tn) - (fp * fn)
        denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + self.epsilon
        return 1 - (num / denom)
