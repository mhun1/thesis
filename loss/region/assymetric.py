from copy import deepcopy

import torch
from torch import nn

from greedy.explore import get_action


class Asymmetric(nn.Module):
    """"
    Implementation based on: https: // ieeexplore.ieee.org / stamp / stamp.jsp?tp = & arnumber = 8759329
    """

    def __init__(self, beta=1.52, apply_non_lin=True):
        super(Asymmetric, self).__init__()

        self.weights = {"beta":beta}
        self.apply_non_lin = apply_non_lin
        self.epsilon = 1e-05

    def apply_weight(self, x, y, action):
        tmp_weights = deepcopy(self.weights)
        num = 0
        for key in tmp_weights.keys():
            val = get_action(action[num])
            tmp_weights[key] = val * self.weights[key]
            num += 1
        return self.forward(x, y, tmp_weights)

    def update(self, action):

        num = 0
        for key in self.weights.keys():
            val = get_action(action[num])
            self.weights[key] = val * self.weights[key]
            num += 1


    def forward(self, x, y, weights=None):

        w_ = self.weights
        if weights:
            w_ = weights

        if self.apply_non_lin:
            x = x.sigmoid()

        tp = torch.sum(x * y)
        fp = torch.sum(x * (1 - y))
        fn = torch.sum((1 - x) * y)
        num = (1 + w_["beta"] ** 2) * tp
        denom = (1 + w_["beta"] ** 2) * tp + (w_["beta"] ** 2) * fn + fp
        return 1 - (num / denom)
