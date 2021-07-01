import torch
from torch import nn
from deap_tools.operations import Scalar


class Dice(nn.Module):
    def __init__(self, smooth=1e-3, apply_nonlin=True, clip=False, evolution=False):
        super(Dice, self).__init__()
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin
        self.clip = clip
        self.evolution = evolution

    def forward(self, x, y):

        if self.evolution:
            x, y = x.val, y.val

        if self.apply_nonlin:
            x = torch.sigmoid(x)

        if self.clip:
            x = torch.clip(x, 0, 1)

        intersection = 2 * torch.sum(x * y) + self.smooth
        cardinal = torch.sum(x) + torch.sum(y) + self.smooth
        dice = 1 - intersection / cardinal

        if self.evolution:
            return Scalar(dice)
        return dice


class LogDice(nn.Module):
    def __init__(self, smooth=1e-3, apply_nonlin=True, clip=False, evolution=False):
        super(LogDice, self).__init__()
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin
        self.clip = clip
        self.evolution = evolution

    def forward(self, x, y):

        if self.evolution:
            x, y = x.val, y.val

        if self.apply_nonlin:
            x = torch.sigmoid(x)

        if self.clip:
            x = torch.clip(x, 0, 1)

        intersection = 2 * torch.sum(x * y) + self.smooth
        cardinal = torch.sum(x) + torch.sum(y) + self.smooth
        dice = -torch.log(intersection/cardinal)

        if self.evolution:
            return Scalar(dice)
        return dice


