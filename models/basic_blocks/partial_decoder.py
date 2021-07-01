import torch
from torch import nn

from models.basic_blocks.conv_block import ConvolutionBlock


class PartialDecoder(nn.Module):
    def __init__(self, channel, dimension):
        super(PartialDecoder, self).__init__()

        mode = "bilinear" if dimension == 2 else "trilinear"
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        self.conv_up1 = ConvolutionBlock(
            channel, channel, dimension, use_activation=False, use_bias=False
        )
        self.conv_up2 = ConvolutionBlock(
            channel, channel, dimension, use_activation=False, use_bias=False
        )
        self.conv_up3 = ConvolutionBlock(
            channel, channel, dimension, use_activation=False, use_bias=False
        )
        self.conv_up4 = ConvolutionBlock(
            channel, channel, dimension, use_activation=False, use_bias=False
        )
        self.conv_up5 = ConvolutionBlock(
            2 * channel, 2 * channel, dimension, use_activation=False, use_bias=False
        )

        self.concat_conv1 = ConvolutionBlock(
            2 * channel, 2 * channel, dimension, use_activation=False, use_bias=False
        )
        self.concat_conv2 = ConvolutionBlock(
            3 * channel, 3 * channel, dimension, use_activation=False, use_bias=False
        )
        self.pre_conv = ConvolutionBlock(
            3 * channel, 3 * channel, dimension, use_activation=False, use_bias=False
        )
        self.out_conv = ConvolutionBlock(
            3 * channel, 1, dimension, kernel_size=1, padding=0, use_activation=False
        )

    def forward(self, x1, x2, x3):
        x_1 = x1
        x_2 = self.conv_up1(self.up(x_1)) * x2
        x_3 = self.conv_up2(self.up(self.up(x1))) * self.conv_up3(self.up(x2)) * x3

        x_2_2 = torch.cat((x_2, self.conv_up4(self.up(x_1))), dim=1)
        x_2_2 = self.concat_conv1(x_2_2)

        x_3_2 = torch.cat((x_3, self.conv_up5(self.up(x_2_2))), dim=1)
        x_3_2 = self.concat_conv2(x_3_2)

        x = self.pre_conv(x_3_2)
        return self.out_conv(x)
