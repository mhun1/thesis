from copy import deepcopy

import torch
from torch import nn
from greedy.explore import get_action

class Effectiveness(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, apply_non_lin=True):
        super(Effectiveness, self).__init__()
        self.weights = {"alpha": alpha, "beta": beta, "gamma":gamma}
        self.apply_non_lin = apply_non_lin
        self.epsilon = 1e-04

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

        TP = torch.sum(x * y)
        FP = torch.sum(x * (1 - y))
        FN = torch.sum((1 - x) * y)

        p = TP/(TP+w_["alpha"]*FP)
        r = TP/(TP+w_["gamma"]*FN)
        return 1 - (1+(w_["beta"])) / (1/p + (w_["beta"])/r)
