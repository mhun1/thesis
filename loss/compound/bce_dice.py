import torch
import torch.nn as nn
import torch.nn.functional as F

from deap_tools.operations import Scalar
from loss.region.dice import Dice


class BCEDice(nn.Module):
    def __init__(
        self, weight_bce=1, weight_dice=1, apply_non_lin=True, evolution=False, device="cuda"
    ):
        super(BCEDice, self).__init__()

        self.weights = {"alpha": 1.0, "beta": 1.0}
        self.losses = {"bce": nn.BCEWithLogitsLoss(pos_weight=torch.full([1], 5, device=device)), "dice": Dice()}
        self.len = len(self.weights)
        self.evolution = evolution

    def fake_forward(self,x,y):
        tmp = torch.zeros([self.len])
        count = 0
        for k,v in self.losses.items():
            tmp[count] = v(x,y)
            count+=1
        return tmp

    def forward(self, x, y):
        if self.evolution:
            x, y = x.val, y.val
        return self.weights["alpha"] * self.losses["bce"](x,y) + self.weights["beta"] * self.losses["dice"](x,y)


class BCE(nn.Module):
    def __init__(self, evolution=False):
        super(BCE, self).__init__()
        self.evolution = evolution

    def forward(self, x, y):
        if self.evolution:
            x, y = x.val, y.val

        result = F.binary_cross_entropy_with_logits(x, y)
        if self.evolution:
            return Scalar(result)
        return F.binary_cross_entropy_with_logits(x, y)
