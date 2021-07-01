import torch
from torchmetrics import Metric
from medpy import metric

class Hausdorff(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("num", default=torch.tensor(0,dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0,dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = preds.clone(), target.clone()

        hd = -1
        try:
            hd = metric.hd(preds.numpy(), target.numpy())
        except:
            hd = -1

        self.num += torch.tensor(1)
        self.total += torch.tensor(hd)

    def compute(self):
        return self.total / self.num




