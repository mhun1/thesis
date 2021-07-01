import torch
import torch.nn.functional as F
from torch import nn

from models.basic_blocks.conv_block import ConvolutionBlock


class ReverseAttention(nn.Module):
    def __init__(
        self,
        dimension,
        in_channels,
        out_channel,
        scale_input,
        scale_map,
        kernel_size=3,
        padding=1,
        last_block=False,
    ):
        super(ReverseAttention, self).__init__()
        self.scale_input = scale_input
        self.scale_map = scale_map

        self.dim = dimension
        self.mode = "bilinear" if dimension == 2 else "trilinear"
        self.in_channels = in_channels
        self.last_block = last_block

        self.conv1 = ConvolutionBlock(
            in_channels,
            out_channel,
            dimension,
            kernel_size=1,
            padding=0,
            use_activation=False,
        )

        self.conv2 = ConvolutionBlock(
            out_channel,
            out_channel,
            dimension,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=False,
        )
        self.conv3 = ConvolutionBlock(
            out_channel,
            out_channel,
            dimension,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=False,
        )

        self.conv4 = ConvolutionBlock(
            out_channel,
            1,
            dimension,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=False,
        )

        if last_block:
            self.conv4 = ConvolutionBlock(
                out_channel,
                out_channel,
                dimension,
                kernel_size=kernel_size,
                padding=padding,
                use_bias=False,
            )
            self.conv5 = ConvolutionBlock(
                out_channel,
                1,
                dimension,
                kernel_size=1,
                padding=0,
                use_activation=False,
                use_bias=False,
            )

    def forward(self, input, x_crop):
        crop = F.interpolate(input, scale_factor=self.scale_input, mode=self.mode)
        x = -1 * (torch.sigmoid(crop)) + 1
        x = x.expand(-1, self.in_channels, -1, -1).mul(x_crop)  # TODO: 3 dim
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.last_block:
            x = self.conv4(x)
            feat = self.conv5(x)

            x = feat + crop
            lateral_map = F.interpolate(x, scale_factor=self.scale_map, mode=self.mode)
            return x, lateral_map

        feat = self.conv4(x)
        x = feat + crop
        lateral_map = F.interpolate(x, scale_factor=self.scale_map, mode=self.mode)
        return x, lateral_map
