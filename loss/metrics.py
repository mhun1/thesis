import numpy as np
import torch


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(
            tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1
        )
        fp = torch.stack(
            tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1
        )
        fn = torch.stack(
            tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1
        )
        tn = torch.stack(
            tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1
        )

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    tp = torch.sum(tp)
    fp = torch.sum(fp)
    fn = torch.sum(fn)
    tn = torch.sum(tn)

    return tp, fp, fn, tn


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp
