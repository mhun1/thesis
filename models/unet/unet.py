from torch import nn

from models.encoder import Encoder, DownBlock
from models.decoder import Decoder
from models.basic_blocks.conv_block import ConvolutionBlock


class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        dimension,
        filter_size=64,
        depth=5,
        bilinear=True,
        use_attention=False,
        use_residual=False,
        use_squeeze=False,
    ):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigmoid = nn.Sigmoid()
        self.dimension = dimension
        self.depth = depth

        self.last_size = 2 ** (depth - 1) * filter_size

        self.encoder = Encoder(
            self.dimension,
            use_residual=use_residual,
            depth=depth,
            use_squeeze=use_squeeze,
        )
        self.decoder = Decoder(
            self.dimension,
            depth=depth,
            bilinear=bilinear,
            use_attention=use_attention,
            use_residual=use_residual,
            use_squeeze=use_squeeze,
        )
        self.last_encoding = DownBlock(
            self.last_size // 2,
            self.last_size,
            self.dimension,
            downsample=False,
            use_residual=use_residual,
            use_squeeze=use_squeeze,
        )
        self.classifier = ConvolutionBlock(
            filter_size,
            n_channels,
            self.dimension,
            use_batch_norm=False,
            use_activation=False,
        )

    def forward(self, x):
        x, skip = self.encoder(x)
        x = self.last_encoding(x)
        x = self.decoder(x, skip)
        x = self.classifier(x)
        return x


