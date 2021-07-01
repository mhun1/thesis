import torch
from medpy import metric

from loss.metrics import get_tp_fp_fn_tn



def accuracy(tn, fp, fn, tp):
    num = tp + tn
    denom = tp + fp + fn + tn
    return num / denom if denom > 0 else 0


def dice(tp, fp, fn):
    num = 2 * tp
    denom = 2 * tp + fp + fn
    return num / denom if denom > 0 else 0


def jaccard(tp, fp, fn):
    num = tp
    denom = tp + fp + fn
    return num / denom if denom > 0 else 0


def precision(tp, fp):
    num = tp
    denom = tp + fp
    return num / denom if denom > 0 else 0


def recall(tp, fn):
    num = tp
    denom = tp + fn
    return num / denom if denom > 0 else 0


def sensitivity(tp, fn):
    num = tp
    denom = tp + fn
    return num / denom if denom > 0 else 0


def specificity(tn, fp):
    num = tn
    denom = fp + tn
    return num / denom if denom > 0 else 0


def f1_score(tp, fp, fn):
    num = tp
    denom = tp + 0.5 * (fp + fn)
    return num / denom if denom > 0 else 0


def f2_score(tp, fp, fn):
    num = 5 * tp
    denom = 5 * tp + 4 * fn + fp
    return num / denom if denom > 0 else 0


def get_eval_metrics(tn, fp, fn, tp):
    out = {
        "accuracy": accuracy(tn, fp, fn, tp),
        "dice": dice(tp, fp, fn),
        "jaccard": jaccard(tp, fp, fn),
        "precision": precision(tp, fp),
        "recall": recall(tp, fn),
        "sensitivity": sensitivity(tp, fn),
        "specificity": specificity(tn, fp),
        "f2_score": f2_score(tp, fp, fn),
    }
    return out


def hd(pred, label):
    hd = -1
    try:
        hd = metric.hd(pred.numpy(), label.numpy())
    except:
        return hd
    return hd


def generate_metrics(x, y, pred):
    if isinstance(pred, list):
        shp_x = x.shape
        axes = list(range(2, len(shp_x)))
        pred_cpu = pred[3].detach().cpu()
        pred_softmax = pred_cpu.sigmoid().data

        res = (pred_cpu - pred_cpu.min()) / (pred_cpu.max() - pred_cpu.min() + 1e-8)
        copy_softmax = torch.where(res < 0.5, 0.0, 1.0)
        tp, fp, fn, tn = get_tp_fp_fn_tn(copy_softmax, y, axes)

        metrics = get_eval_metrics(tn, fp, fn, tp)
        metrics["img"] = x
        metrics["label"] = y
        metrics["pred"] = pred_cpu
        metrics["softmax"] = pred_softmax
        metrics["threshold"] = copy_softmax
        metrics["hd"] = hd(copy_softmax, y)
        return metrics
    else:
        pred = pred.detach()
        shp_x = x.shape
        axes = list(range(2, len(shp_x)))
        pred_softmax = torch.sigmoid(pred).cpu()  # [B,C,W,H]
        pred_cpu = pred.cpu()
        copy_softmax = torch.where(pred_softmax < 0.5, 0.0, 1.0)

        tp, fp, fn, tn = get_tp_fp_fn_tn(copy_softmax, y, axes)
        metrics = get_eval_metrics(tn, fp, fn, tp)

        # convert to to float32 cause matplotlib cant handle float16
        # https://github.com/matplotlib/matplotlib/issues/15432

        metrics["img"] = x.to(torch.float32)
        metrics["label"] = y.to(torch.float32)
        metrics["pred"] = pred_cpu.to(torch.float32)
        metrics["softmax"] = pred_softmax.to(torch.float32)
        metrics["threshold"] = copy_softmax.to(torch.float32)
        metrics["hd"] = hd(copy_softmax, y)
        return metrics

def generate_metrics_greedy(x, y, pred):
    pred = pred.detach()
    shp_x = x.shape
    axes = list(range(2, len(shp_x)))
    pred_softmax = torch.sigmoid(pred).cpu()  # [B,C,W,H]
    pred_cpu = pred.cpu()
    copy_softmax = torch.where(pred_softmax < 0.5, 0.0, 1.0)

    tp, fp, fn, tn = get_tp_fp_fn_tn(copy_softmax, y, axes)
    metrics = get_eval_metrics(tn, fp, fn, tp)
    metrics["hd"] = hd(copy_softmax, y)
    return metrics


def avg_val(outputs, key):
    return sum([x[key] for x in outputs]) / len([x[key] for x in outputs])

