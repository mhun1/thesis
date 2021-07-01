import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from deap_tools.operations import Scalar


class Focal(nn.Module):
    def __init__(self, gamma=1.45, alpha=0.25, evolution=False, reduction="mean"):
        super(Focal, self).__init__()
        self.gamma = gamma
        self.alpha = 1 - alpha
        self.evolution = evolution
        if reduction in ("mean", "sum", "none"):
            self.reduction = reduction
        else:
            raise ValueError("Choose from: mean,sum,none")

    def forward(self, x: Tensor, y: Tensor):
        if self.evolution:
            x, y = x.val, y.val

        ce = F.binary_cross_entropy_with_logits(x, y, reduction="mean")
        pt = torch.exp(-ce)
        fl = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            result = fl.mean()
        elif self.reduction == "sum":
            result = fl.sum()
        else:
            raise NotImplementedError("Not implemented")
        if self.evolution:
            return Scalar(result)
        return result

