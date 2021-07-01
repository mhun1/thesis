import torch
import torch.nn as nn
from scipy import ndimage
from torch import einsum

from datasets.cervical import dataset_cervical
from datasets.transforms import get_transforms
from loss.region.dice import Dice


class PureSurface(nn.Module):
    def __init__(
        self,
        apply_non_lin=True,
        device="cuda",
        normalized=False,
    ):
        """
        :param reduction:
        :param transform:
        :param apply_non_lin:
        """
        super(PureSurface, self).__init__()
        self.apply_non_lin = apply_non_lin
        self.device = device

    def forward(self, x, y):
        if self.apply_non_lin:
            x = x.sigmoid().to(self.device)

        y = torch.from_numpy(
            ndimage.distance_transform_edt(torch.logical_not(y.cpu()))
        ).to(self.device)

        return (x * y).mean()

class Surface(nn.Module):
    def __init__(
        self,
    ):
        super(Surface, self).__init__()
        self.weights = {"alpha": 0.5, "beta": 1.0}
        self.losses = {"dce": Dice(apply_nonlin=False), "boundary": PureSurface(apply_non_lin=False)}
        self.len = len(self.losses.keys())

    def fake_forward(self,x,y):
        tmp = torch.zeros([self.len])
        count = 0
        for k,v in self.losses.items():
            tmp[count] = v(x,y)
            count+=1
        return tmp

    def forward(self, x, y):
        x = x.sigmoid()
        return self.weights["beta"] * self.losses["dce"](x,y) + self.weights["alpha"] * self.losses["boundary"](x,y)

class RebalanceSurface(nn.Module):
    def __init__(
            self,
    ):
        super(RebalanceSurface, self).__init__()
        self.losses = {"dce": Dice(apply_nonlin=False), "boundary": PureSurface(apply_non_lin=False)}
        self.alpha = 0.01

    def forward(self, x, y):
        x = x.sigmoid()
        loss = (1-self.alpha)*self.losses["dce"](x,y) + self.alpha*self.losses["boundary"](x,y)
        self.alpha += 0.005
        return loss






