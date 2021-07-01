import torch
from torch import nn

from models.basic_blocks.attention import AttentionBlock
from models.basic_blocks.conv_block import ConvolutionBlock
from models.utils.torch_utils import get_function, get_up
from models.basic_blocks.se import SE


class Decoder(nn.Module):
    def __init__(
        self,
        dimension,
        n_channels=1,
        depth=5,
        use_attention=False,
        use_residual=False,
        use_squeeze=False,
        bilinear=False,
        double_unet=False,
        second_decoder=False,
    ):
        super(Decoder, self).__init__()

        self.decode = nn.ModuleList()
        val = 2 ** (depth - 1) * 64

        is_first = 0
        for _ in range(1, depth):

            if double_unet and is_first == 0:
                self.decode.append(
                    UpBlock(
                        val,
                        val // 2,
                        dimension,
                        bilinear=bilinear,
                        use_residual=use_residual,
                        conv_up=True,
                        second_decoder=second_decoder,
                        use_squeeze=use_squeeze,
                    )
                )
                is_first += 1
            else:
                if second_decoder:
                    self.decode.append(
                        UpBlock(
                            val,
                            val // 2,
                            dimension,
                            use_attention=use_attention,
                            bilinear=bilinear,
                            use_residual=use_residual,
                            second_decoder=second_decoder,
                            use_squeeze=use_squeeze,
                        )
                    )
                else:

                    self.decode.append(
                        UpBlock(
                            val,
                            val // 2,
                            dimension,
                            use_attention=use_attention,
                            bilinear=bilinear,
                            use_residual=use_residual,
                            use_squeeze=use_squeeze,
                        )
                    )
            val //= 2

    def forward(self, x, skip_connections):

        is_tuple = False
        if isinstance(skip_connections[0], tuple):
            is_tuple = True

        compact = zip(reversed(skip_connections), self.decode)
        for skip, up in compact:
            if is_tuple:
                x = up(x, torch.cat((skip[0], skip[1]), dim=1))
            else:
                x = up(x, skip)
        return x


class UpBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        dimension,
        use_attention=False,
        use_residual=False,
        bilinear=False,
        conv_up=False,
        second_decoder=False,
        use_squeeze=False,
    ):

        super().__init__()

        self.conv1 = None
        if second_decoder or bilinear:
            self.conv1 = ConvolutionBlock(
                in_channels + in_channels // 2, out_channels, dimension
            )
        else:
            self.conv1 = ConvolutionBlock(in_channels, out_channels, dimension)

        self.conv2 = ConvolutionBlock(out_channels, out_channels, dimension)

        self.residual = None
        if use_residual:

            tmp_channels = in_channels
            if bilinear:
                tmp_channels = in_channels + in_channels // 2

            self.residual = ConvolutionBlock(
                tmp_channels,
                out_channels,
                dimension,
                kernel_size=1,
                padding=0,
                padding_mode="zeros",
                use_batch_norm=True,
                use_activation=True,
            )

        if bilinear:
            self.up = get_up(dimension)

        else:
            ##TODO: DOUBLE UNET CHANGE IN_CHANNELS TO 64 and CHANGE!

            if conv_up:
                self.up = get_function(nn, "ConvTranspose{}d".format(dimension))(
                    64, 512, kernel_size=2, stride=2
                )
            else:
                self.up = get_function(nn, "ConvTranspose{}d".format(dimension))(
                    in_channels, in_channels // 2, kernel_size=2, stride=2
                )

        attn_channel = in_channels if bilinear else in_channels // 2

        self.attention = (
            AttentionBlock(attn_channel, in_channels // 2, out_channels, dimension)
            if use_attention
            else None
        )
        self.squeeze_block = SE(out_channels, dimension) if use_squeeze else None

    def forward(self, x, skip):
        x = self.up(x)

        if self.attention:
            x = self.attention(x, skip)

        x = torch.cat((skip, x), dim=1)
        if self.residual:

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
        return x
