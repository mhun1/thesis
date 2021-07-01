import torch
from torch import nn

from models.basic_blocks.conv_block import ConvolutionBlock


class AttentionBlock(nn.Module):
    def __init__(self, in_x, in_g, out, dimension):
        super(AttentionBlock, self).__init__()

        self.W_x = ConvolutionBlock(
            in_x,
            out,
            dimension=dimension,
            kernel_size=1,
            padding=0,
            use_bias=True,
            use_activation=False,
        )

        self.W_g = ConvolutionBlock(
            in_g,
            out,
            dimension=dimension,
            kernel_size=1,
            padding=0,
            use_bias=True,
            use_activation=False,
        )

        self.psi = ConvolutionBlock(
            out,
            1,
            dimension=dimension,
            kernel_size=1,
            padding=0,
            use_bias=True,
            activation="Sigmoid",
        )

        self.relu = nn.ReLU()

    def forward(self, x, g):
        g = self.W_g(g)
        x_1 = self.W_x(x)
        psi = self.relu(g + x_1)
        return x * self.psi(psi)


#
# x = torch.ones((1,128,128,128))
# g = torch.ones((1,128,128,128))
#
# attention = AttentionBlock(128,128,1,2)
#
# print(attention(x,g).shape)
