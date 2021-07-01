import torch
from torch import nn

from loss.distribution.focal import Focal
from loss.region.dice import Dice


class DiceFocal(nn.Module):
    def __init__(self, gamma=4/3, apply_non_lin=True):
        super(DiceFocal, self).__init__()
        self.dice = Dice(apply_nonlin=apply_non_lin)
        self.focal = Focal()
        self.gamma = gamma

    def forward(self, x, y):
        return self.dice(x,y) + self.focal(x,y)