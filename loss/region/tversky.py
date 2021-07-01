from copy import deepcopy

import torch
from torch import nn
from greedy.explore import get_action


class Tversky(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, apply_non_lin=True, device="cuda"):
        super(Tversky, self).__init__()
        self.weights = {"alpha": torch.tensor(alpha, device= device), "beta": torch.tensor(beta,device= device)}
        self.apply_non_lin = apply_non_lin
        self.epsilon = 1e-04
        self.softmax = nn.Softmax()

    def apply_weight(self, x, y, action):
        tmp_weights = deepcopy(self.weights)
        num = 0
        for key in tmp_weights.keys():
            val = get_action(action[num])
            tmp_weights[key] = val * tmp_weights[key]
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
        return 1 - (TP + self.epsilon) / (
            TP + w_["alpha"] * FP + w_["beta"] * FN + self.epsilon
        )

