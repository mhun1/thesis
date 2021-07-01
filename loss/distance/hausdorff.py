from copy import deepcopy

import torch
from torch import nn
from scipy import ndimage
from loss.region.dice import Dice


def dt(z):
    x = deepcopy(z.cpu().detach())
    dt = torch.zeros_like(x, device="cpu")
    for idx in range(dt.shape[0]):
        foreground = x[idx] > 0.5
        if foreground.any():
            background = ~foreground
            f_dist = torch.from_numpy(ndimage.distance_transform_edt(foreground.numpy())).to("cpu")
            b_dist = torch.from_numpy(ndimage.distance_transform_edt(background.numpy())).to("cpu")
            dt[idx] = f_dist + b_dist
    return dt

class PureHausdorff(nn.Module):
    def __init__(self):
        super(PureHausdorff, self).__init__()

    def forward(self,x,y):
        dt_x = dt(x)
        dt_y = dt(y)
        err = (x - y) ** 2
        dist = dt_x ** 2 + dt_y ** 2
        hd = (err * dist).mean()
        return hd

class Hausdorff(nn.Module):
    def __init__(self, rebalance=False):
        super(Hausdorff, self).__init__()
        self.dice = Dice(apply_nonlin=False)
        self.weights = {"alpha": 1.0, "beta": 0.3}
        self.hd = PureHausdorff()
        self.losses = {"alpha": self.dice, "beta": self.hd}
        self.len = len(self.losses)


    def fake_forward(self,x,y):
        tmp = torch.zeros([self.len])
        count = 0
        h_ = x.clone().sigmoid()
        for k,v in self.losses.items():
            tmp[count] = v(h_,y)
            count+=1
        return tmp

    def forward(self, x, y):
        x = x.sigmoid()
        loss = self.weights["beta"] * self.losses["alpha"](x,y) + self.weights["alpha"]*self.losses["beta"](x,y)
        return loss

class RebalanceHausdorff(nn.Module):
    def __init__(self):
        super(RebalanceHausdorff, self).__init__()
        self.dice = Dice(apply_nonlin=False)
        self.weights = {"alpha": 1.0, "beta": 0.3}
        self.hd = PureHausdorff()
        self.losses = {"alpha": self.dice, "beta": self.hd}
        self.len = len(self.losses)
        self.alpha = 0.01

    def forward(self, x, y):
        x = x.sigmoid()
        loss = (1-self.alpha)*self.losses["alpha"](x,y) + self.alpha*self.losses["beta"](x,y)
        self.alpha += 0.0003 # 0.0003 for Cervical # 0.0001 for Lumbar
        return loss

