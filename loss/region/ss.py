import torch
from torch import nn


class SS(nn.Module):

    """
    Implementation based on: https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, lmbda=0.05, apply_non_lin=True):
        super(SS, self).__init__()
        self.lmbda = lmbda
        self.apply_non_lin = apply_non_lin
        self.epsilon = 1e-05

    def forward(self, x, y):
        if self.apply_non_lin:
            x = x.sigmoid()

        sensitivity = (torch.sum((y - x) ** 2 * y)) / (torch.sum(y) + self.epsilon)
        specificity = (torch.sum((y - x) ** 2 * (1 - y))) / (
            torch.sum(1 - y) + self.epsilon
        )
        return self.lmbda * sensitivity + (1 - self.lmbda) * specificity
