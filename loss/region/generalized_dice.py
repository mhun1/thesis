import torch
from torch import nn
from torch import einsum


class GeneralizedDice(nn.Module):
    """
    Implementation based on: https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, apply_non_lin=True):
        super(GeneralizedDice, self).__init__()
        self.epsilon = 1e-07
        self.apply_non_lin = apply_non_lin

    def forward(self, x, y):
        # SHAPE: (B,C,Z,W,H)
        if self.apply_non_lin:
            x = x.sigmoid()

        w = 1 / ((torch.sum(y) + self.epsilon) ** 2)
        intersection = w * torch.sum(x * y)
        union = w * (torch.sum(x) + torch.sum(y))
        divided = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        loss = divided.mean()
        return loss

