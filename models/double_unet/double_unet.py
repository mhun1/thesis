import torch

from torch import nn
from models.encoder import Encoder, DownBlock
from models.decoder import Decoder
from models.basic_blocks.conv_block import ConvolutionBlock
from models.basic_blocks.aspp import ASPP, OutBlock


class DoubleUNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        dimension,
        filter_size=64,
        depth=5,
        bilinear=True,
        residual=False,
    ):
        super(DoubleUNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sigmoid = nn.Sigmoid()
        self.dimension = dimension
        self.depth = depth

        self.last_size = 2 ** (depth - 1) * filter_size

        self.encoder_1 = Encoder(
            self.dimension, use_residual=residual, depth=depth, use_first_block=True
        )
        self.decoder_1 = Decoder(
            self.dimension,
            use_residual=residual,
            depth=depth,
            bilinear=bilinear,
            double_unet=True,
        )

        self.encoder_2 = Encoder(self.dimension, use_residual=residual, depth=depth)
        self.decoder_2 = Decoder(
            self.dimension,
            use_residual=residual,
            depth=depth,
            bilinear=bilinear,
            double_unet=True,
            second_decoder=True,
        )

        self.last_encoding = DownBlock(
            self.last_size // 2,
            self.last_size,
            self.dimension,
            downsample=False,
            use_residual=residual,
        )

        self.last_encoding_2 = DownBlock(
            self.last_size // 2,
            self.last_size,
            self.dimension,
            downsample=False,
            use_residual=residual,
        )

        self.classifier = ConvolutionBlock(
            filter_size,
            n_channels,
            self.dimension,
            use_batch_norm=False,
            use_activation=False,
        )

        self.out_block = OutBlock(64, self.dimension)
        self.aspp = ASPP(inputs=1024, filter=64, dimension=self.dimension)

    def forward(self, x):

        input = x
        x, skip, out = self.encoder_1(x)
        x = self.last_encoding(x)
        # print(out.shape)
        # print("X: AFTER ENCODING: ", x.shape)
        x = self.aspp(x)
        # print("AFTER ASPP: ", x.shape)
        x = self.decoder_1(x, skip)
        out_1 = self.out_block(x)
        # print("OUT: ", out_1.shape)
        x = input * out_1
        # print("OUT: ", x.shape)
        x, skip_2 = self.encoder_2(x)
        x = self.last_encoding_2(x)
        # print("BEFORE ASPP: ", x.shape)
        x = self.aspp(x)
        skips = tuple(zip(skip, skip_2))
        x = self.decoder_2(x, skips)
        out_2 = self.out_block(x)
        final_out = torch.cat((out_1, out_2), dim=1)
        # print(final_out.shape)

        return out_2


# x = torch.Tensor(1,1,64,128,128)
# net = DoubleUNet(1,1,3,bilinear=False,depth=5)
# #
# out = net(x)
# print(out.shape)
# print("FINISHED")
#
# pytorch_total_params = sum(p.numel() for p in net.parameters())
# print(pytorch_total_params)
