import torch

def get_components(x, y):
    x = x.sigmoid().detach()
    y = y.detach()
    TP = torch.sum(x * y).cpu().numpy()
    TN = torch.sum((1 - x) * (1 - y)).cpu().numpy()
    FP = torch.sum(x * (1 - y)).cpu().numpy()
    FN = torch.sum((1 - x) * y).cpu().numpy()
    return TP, TN, FP, FN


def p_n_blocks(x, y, normalized=True):
    TP = torch.sum(x * y)
    TN = torch.sum((1 - x) * (1 - y))
    FP = torch.sum(x * (1 - y))
    FN = torch.sum((1 - x) * y)
    sum = TP + TN + FP + FN
    TP = TP / sum
    TN = TN / sum
    FP = FP / sum
    FN = FN / sum
    return TP, TN, FP, FN
