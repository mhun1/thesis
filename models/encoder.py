from torch import nn
from models.basic_blocks.conv_block import ConvolutionBlock
from models.basic_blocks.se import SE


class Encoder(nn.Module):
    def __init__(
        self,
        dimension,
        n_channels=1,
        depth=5,
        use_residual=False,
        use_first_block=False,
        use_squeeze=False,
    ):
        super(Encoder, self).__init__()

        self.encode = nn.ModuleList()
        self.need_first_block = use_first_block

        val = 64
        first_block = True
        for _ in range(1, depth):
            if first_block:

                self.encode.append(
                    DownBlock(
                        n_channels,
                        val,
                        dimension,
                        use_residual=use_residual,
                        use_squeeze=use_squeeze,
                    )
                )
                first_block = False
            else:

                self.encode.append(
                    DownBlock(
                        val,
                        2 * val,
                        dimension,
                        use_residual=use_residual,
                        use_squeeze=use_squeeze,
                    )
                )
                val *= 2

    def forward(self, x):
        skip_connections = []
        if self.need_first_block:
            count = 0
            for block in self.encode:
                x, skip = block(x)
                if count == 0:
                    out = x
                skip_connections.append(skip)
                count += 1
            return x, skip_connections, out
        else:
            for block in self.encode:
                x, skip = block(x)
                skip_connections.append(skip)
            return x, skip_connections


class DownBlock(nn.Module):
    """Downscaling with MaxPooling -> DoubleConv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        dimension,
        use_residual=False,
        downsample=True,
        use_squeeze=False,
    ):
        super().__init__()

        self.downsample = downsample
        self.conv1 = ConvolutionBlock(in_channels, out_channels, dimension)
        self.conv2 = ConvolutionBlock(out_channels, out_channels, dimension)

        self.use_residual = use_residual
        if self.use_residual:
            self.residual = ConvolutionBlock(
                in_channels,
                out_channels,
                dimension,
                kernel_size=1,
                padding=0,
                padding_mode="zeros",
                use_batch_norm=True,
                use_activation=True,
            )

        self.squeeze_block = None
        if use_squeeze:
            self.squeeze_block = SE(out_channels, dimension)

        if dimension == 2:
            self.down = nn.MaxPool2d(2)
        elif dimension == 3:
            self.down = nn.MaxPool3d(2)

    def forward(self, x):
        if self.use_residual:
            res = self.residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            if self.squeeze_block:
                x = self.squeeze_block(x)
            x += res
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            if self.squeeze_block:
                x = self.squeeze_block(x)

        if not self.downsample:
            return x

        else:
            skip = x
            x = self.down(x)
            return x, skip
