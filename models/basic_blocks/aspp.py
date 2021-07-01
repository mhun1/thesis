import torch
import torch.nn as nn
from models.basic_blocks.conv_block import ConvolutionBlock
from models.utils.torch_utils import get_function


class ASPP(nn.Module):
    def __init__(self, inputs=1024, filter=64, dimension=3):
        super(ASPP, self).__init__()

        self.filter = filter
        self.avg = get_function(nn, "AvgPool{}d".format(dimension))((1))

        mode = "bilinear"
        if dimension == 3:
            mode = "trilinear"

        self.up = nn.Upsample(mode=mode)
        self.init_conv = ConvolutionBlock(
            inputs, filter, dimension=dimension, kernel_size=1, padding=0, dilation=1
        )

        self.conv1 = ConvolutionBlock(
            inputs, filter, dimension=dimension, kernel_size=1, padding=0, dilation=1
        )

        self.conv2 = ConvolutionBlock(
            inputs, filter, dimension=dimension, kernel_size=3, padding=6, dilation=6
        )

        self.conv3 = ConvolutionBlock(
            inputs, filter, dimension=dimension, dilation=12, padding=12, kernel_size=3
        )

        self.conv4 = ConvolutionBlock(
            inputs, filter, dimension=dimension, dilation=18, padding=18, kernel_size=3
        )

        self.out = ConvolutionBlock(
            self.filter * 4, 64, dimension=dimension, kernel_size=1, padding=0
        )

    def forward(self, x):
        # print(x.shape)

        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_3 = self.conv3(x)
        x_4 = self.conv4(x)
        x_5 = self.avg(x)

        # print("X_1: ", x_1.shape)
        # print("X_2: ", x_2.shape)
        # print("X_3: ", x_3.shape)
        # print("X_4: ", x_4.shape)

        out = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        # print("OUT: ", out.shape)
        # print()

        out = self.out(out)
        # print(out.shape)
        return out


class OutBlock(nn.Module):
    def __init__(self, filter_size, dimension):
        super(OutBlock, self).__init__()
        self.conv = ConvolutionBlock(
            filter_size,
            1,
            dimension=dimension,
            use_batch_norm=False,
            activation="Sigmoid",
        )

    def forward(self, x):
        return self.conv(x)


# x = torch.ones((1,5,32,32))
# y = torch.ones((1,5,32,32))
#
# #print(torch.cat((x,y)).shape)
#
# #avg = nn.AvgPool2d((x.shape[0],x.shape[1]))
# #print(avg(x).shape)
#
# #out = OutBlock(5,2)
# #print(out(x).shape)
#
# x_list = [x,x,x]
# y_list = [y,y,y]
#
# tab = tuple(zip(x_list,y_list))
# print(torch.cat((tab[0][0],tab[0][1])).shape)
