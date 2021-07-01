from torchmetrics import MetricCollection, Accuracy, F1, FBeta, IoU, Precision, Recall
from utils.hd_metric import Hausdorff

def get_metrics():
    return MetricCollection(
        [
            Accuracy(),
            F1(),
            FBeta(beta=2),
            IoU(2, reduction="none"),
            Precision(),
            Recall(),
            Hausdorff(),
        ]
    )