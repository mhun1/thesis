import torch
import torch.nn as nn

from deap_tools.operations import Scalar


class ActiveContour(nn.Module):

    """
    Implementation of the Active Contour Loss defined in:
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf

    Note:
    For training stabilization purposes this loss should be applied on pre-trained weights
    (e.g first epochs with cross entropy loss)

    INPUT SHAPES: [B,C,(Z),W,H]
    """

    def __init__(
        self, evolution=False, reduction="mean", region_weight=1, contour_weight=1
    ):
        super(ActiveContour, self).__init__()
        self.epsilon = 1e-6
        self.reduction = reduction
        self.region_weight = (
            region_weight  # TODO: couldnt read exact value out of paper between 0-10
        )
        self.contour_weight = contour_weight
        self.evolution = evolution

    def forward(self, x, y):
        if self.evolution:
            x, y = x.val, y.val

        if len(x.shape) == 3:
            x = x.unsqueeze(0).unsqueeze(0)
            y = y.unsqueeze(0).unsqueeze(0)

        horizontal_grad = None
        vertical_grad = None

        if len(x.shape) - 2 == 2:
            "shape: [B,C,W,H]"
            horizontal_grad = x[:, :, 1:, :] - x[:, :, :-1, :]
            vertical_grad = x[:, :, :, 1:] - x[:, :, :, :-1]

            horizontal_grad = horizontal_grad[:, :, 1:, :-2] ** 2
            vertical_grad = vertical_grad[:, :, :-2, 1:] ** 2

        if len(x.shape) - 2 == 3:
            "shape: [B,C,Z,W,H]"
            horizontal_grad = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
            vertical_grad = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

            horizontal_grad = horizontal_grad[:, :, :, 1:, :-2] ** 2
            vertical_grad = vertical_grad[:, :, :, :-2, 1:] ** 2

        contour = torch.abs(horizontal_grad + vertical_grad)
        region = 0

        if self.reduction == "mean":
            contour = torch.mean(torch.sqrt(contour + self.epsilon))
            inner_region = torch.mean(y * (x - torch.ones_like(x)) ** 2)
            outer_region = torch.mean((1 - y) * (x - torch.zeros_like(x)) ** 2)
            region = inner_region + outer_region

        elif self.reduction == "sum":
            contour = torch.sum(torch.sqrt(contour + self.epsilon))
            inner_region = torch.sum(y * (x - torch.ones_like(x)) ** 2)
            outer_region = torch.sum((1 - y) * (x - torch.zeros_like(x)) ** 2)
            region = inner_region + outer_region
        else:
            raise NotImplementedError("Choose reduction type: mean,sum,none")

        result = self.contour_weight * contour + self.region_weight * region
        if self.evolution:
            return Scalar(result)
        return result
