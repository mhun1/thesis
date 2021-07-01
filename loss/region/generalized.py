import collections

import torch
from torch import nn

action_lookup = [0.1, 0.25, 0.5, 0.75, 1.1, 1.25, 1.5, 1.7]


class Generalized(nn.Module):
    def __init__(self, alpha_p=1, alpha_r=1, beta=1, apply_non_lin=True):
        super(Generalized, self).__init__()

        self.apply_non_lin = apply_non_lin
        self.epsilon = 1e-07
        self.val_dict = collections.OrderedDict(
            {
                "alpha_p": alpha_p,  # -> tp / fp
                "alpha_r": alpha_r,  # -> tp / fn
                "beta": beta,  # -> R/P
            }
        )

    def get_vals(self, x, y):
        if self.apply_non_lin:
            x = x.sigmoid().detach()

        y = y.detach()

        tp = torch.sum(x * y) + self.epsilon
        fp = torch.sum(x * (1 - y)) + self.epsilon
        fn = torch.sum((1 - x) * y) + self.epsilon

        prec = tp / (tp + self.val_dict["alpha_p"] * fp)
        rec = tp / (tp + self.val_dict["alpha_r"] * fn)

        out_dict = {"alpha_p": tp / fp, "alpha_r": tp / fn, "beta": rec / prec}
        return out_dict

    def apply_action(self, update):
        # gets tuple (action,num_parameter)
        print(update)
        tup = list(self.val_dict.items())[update[1] - 1]
        self.val_dict[tup[0]] = action_lookup[update[0]] * tup[1]

    def forward(self, x, y, dict=None):
        if self.apply_non_lin:
            x = x.sigmoid()

        if dict:
            self.val_dict.update((x, y * dict[x]) for x, y in self.val_dict.items())

        tp = torch.sum(x * y) + self.epsilon
        fp = torch.sum(x * (1 - y)) + self.epsilon
        fn = torch.sum((1 - x) * y) + self.epsilon

        prec = tp / (tp + self.val_dict["alpha_p"] * fp)
        rec = tp / (tp + self.val_dict["alpha_r"] * fn)
        return 1 - (1 + self.val_dict["beta"] ** 2) / (
            1 / prec + (self.val_dict["beta"] ** 2 / rec)
        )
