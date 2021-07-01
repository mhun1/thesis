import torch
import torch.nn as nn
import torch.nn.functional as F

def get_gt_proportion(labels):

    valid_mask = (labels >= 0.0)
    if labels.dim() == 3:
        labels = labels.unsqueeze(dim=1)
    bin_labels = labels
    gt_proportion = get_region_proportion(bin_labels, valid_mask)
    return gt_proportion, valid_mask

def get_region_proportion(x, valid_mask=None, EPS=1e-5) -> torch.Tensor:
    if valid_mask is not None:
        if valid_mask.dim() == 4:
            x = torch.einsum("bcwh, bcwh->bcwh", x, valid_mask)
            cardinality = torch.einsum("bcwh->bc", valid_mask)
        else:
            x = torch.einsum("bczwh,bczwh->bczwh", x, valid_mask)
            cardinality = torch.einsum("bczwh->bc", valid_mask)
    else:
        cardinality = x.shape[2] * x.shape[3]
    region_proportion = (torch.einsum("bczwh->bc", x) + EPS) / (cardinality + EPS)
    return region_proportion


def get_pred_proportion(logits,temp=1.0,valid_mask=None):
    preds = F.logsigmoid(temp * logits).exp()
    pred_proportion = get_region_proportion(preds, valid_mask)
    return pred_proportion



class BCEL1(nn.Module):
    def __init__(
        self, gamma=1.0
    ):
        super(BCEL1, self).__init__()
        self.gamma = gamma

    def forward(self, x, y):
        bce = F.binary_cross_entropy_with_logits(x, y)
        gt_proportion, valid_mask = get_gt_proportion(y)
        pred_proportion = get_pred_proportion(x, temp=1.0, valid_mask=valid_mask)
        regional = (pred_proportion - gt_proportion).abs().mean()
        loss = bce + self.gamma * regional
        return loss


