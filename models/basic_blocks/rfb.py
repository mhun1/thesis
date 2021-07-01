import torch
from torch import nn
from models.basic_blocks.conv_block import ConvolutionBlock


class RFB_2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RFB_2D, self).__init__()
        dimension = 2
        self.relu = nn.ReLU(True)

        self.conv1 = ConvolutionBlock(
            in_channels,
            out_channels,
            dimension,
            kernel_size=1,
            padding=0,
            use_activation=False,
        )

        self.conv2 = nn.Sequential(
            ConvolutionBlock(
                in_channels,
                out_channels,
                dimension,
                kernel_size=1,
                padding=0,
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(1, 3),
                padding=(0, 1),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(3, 1),
                padding=(1, 0),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=3,
                padding=3,
                dilation=3,
                use_activation=False,
            ),
        )

        self.conv3 = nn.Sequential(
            ConvolutionBlock(
                in_channels,
                out_channels,
                dimension,
                kernel_size=1,
                padding=0,
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(1, 5),
                padding=(0, 2),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(5, 1),
                padding=(2, 0),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=3,
                padding=5,
                dilation=5,
                use_activation=False,
            ),
        )

        self.conv4 = nn.Sequential(
            ConvolutionBlock(
                in_channels,
                out_channels,
                dimension,
                kernel_size=1,
                padding=0,
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(1, 7),
                padding=(0, 3),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(7, 1),
                padding=(3, 0),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=3,
                padding=7,
                dilation=7,
                use_activation=False,
            ),
        )

        self.conv_cat = ConvolutionBlock(
            4 * out_channels, out_channels, dimension, use_activation=False
        )
        self.conv_res = ConvolutionBlock(
            in_channels,
            out_channels,
            dimension,
            padding=0,
            kernel_size=1,
            use_activation=False,
        )

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_3 = self.conv3(x)
        x_4 = self.conv4(x)

        x_cat = self.conv_cat(torch.cat((x_1, x_2, x_3, x_4), dim=1))
        return self.relu(x_cat + self.conv_res(x))


class RFB_3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RFB_3D, self).__init__()
        dimension = 3
        self.relu = nn.ReLU(True)

        self.conv1 = ConvolutionBlock(
            in_channels, out_channels, dimension, kernel_size=1, use_activation=False
        )

        self.conv2 = nn.Sequential(
            ConvolutionBlock(
                in_channels,
                out_channels,
                dimension,
                kernel_size=1,
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(1, 1, 3),
                padding=(0, 0, 1),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(1, 3, 1),
                padding=(0, 1, 0),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=3,
                padding=3,
                dilation=3,
                use_activation=False,
            ),
        )

        self.conv3 = nn.Sequential(
            ConvolutionBlock(
                in_channels,
                out_channels,
                dimension,
                kernel_size=1,
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(1, 1, 5),
                padding=(0, 0, 2),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(1, 5, 1),
                padding=(0, 2, 0),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(5, 1, 1),
                padding=(2, 0, 0),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=3,
                padding=5,
                dilation=5,
                use_activation=False,
            ),
        )

        self.conv4 = nn.Sequential(
            ConvolutionBlock(
                in_channels,
                out_channels,
                dimension,
                kernel_size=1,
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(1, 1, 7),
                padding=(0, 0, 3),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(1, 7, 1),
                padding=(0, 3, 0),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=(7, 1, 1),
                padding=(3, 0, 0),
                use_activation=False,
            ),
            ConvolutionBlock(
                out_channels,
                out_channels,
                dimension,
                kernel_size=3,
                padding=7,
                dilation=7,
                use_activation=False,
            ),
        )

        self.conv_cat = (
            ConvolutionBlock(
                4 * out_channels, out_channels, dimension, use_activation=False
            ),
        )
        self.conv_res = ConvolutionBlock(
            in_channels, out_channels, dimension, kernel_size=1, use_activation=False
        )

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_3 = self.conv3(x)
        x_4 = self.conv4(x)

        x_cat = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        x_cat = self.conv_cat(x_cat)
        return self.relu(x_cat + self.conv_res(x))
