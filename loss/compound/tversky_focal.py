import torch
from torch import nn
from loss.region.tversky import Tversky


class TverskyFocal(nn.Module):
    """"
    Implementation based on: https: // ieeexplore.ieee.org / stamp / stamp.jsp?tp = & arnumber = 8759329
    Authors suggest: alpha = 0.7 | beta 0.3 | gamma = 4/3
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=4 / 3, apply_non_lin=True):
        super(TverskyFocal, self).__init__()
        self.tversky = Tversky(alpha=alpha, beta=beta, apply_non_lin=apply_non_lin)
        self.gamma = gamma

    def forward(self, x, y):
        return self.tversky(x, y) ** (1 / self.gamma)
