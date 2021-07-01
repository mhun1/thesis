import torch
from torch import nn
from models.utils.torch_utils import get_function


class SE(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, dimension, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(SE, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.avg = get_function(nn, "AvgPool{}d".format(dimension))((1))
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        b, c = 0, 0
        if len(x.shape) - 2 == 3:
            b, c, _, _, _ = x.size()
        elif len(x.shape) - 2 == 2:
            b, c, _, _ = x.size()

        # Average along each channel

        squeeze_tensor = self.avg(x)
        squeeze_tensor = squeeze_tensor.view(b, c, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        y = None
        if len(x.shape) - 2 == 2:
            y = fc_out_2.view(b, c, 1, 1)

        if len(x.shape) - 2 == 3:
            y = fc_out_2.view(b, c, 1, 1, 1)

        return torch.mul(x, y)
