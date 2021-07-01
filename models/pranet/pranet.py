import torch
import torch.nn.functional as F

from torch import nn
from models.basic_blocks.conv_block import ConvolutionBlock
from models.backbone.res2net_v1b import res2net50_v1b_26w_4s
from models.basic_blocks.partial_decoder import PartialDecoder
from models.basic_blocks.reverse_attention import ReverseAttention
from models.basic_blocks.rfb import RFB_2D, RFB_3D
from models.utils.torch_utils import get_function


class PraNet(nn.Module):
    def __init__(self, channel, dimension, rgb=False):
        super(PraNet, self).__init__()

        self.rgb = rgb
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.mode = "bilinear" if dimension == 2 else "trilinear"
        if dimension == 2:
            self.rfb_2 = RFB_2D(512, channel)
            self.rfb_3 = RFB_2D(1024, channel)
            self.rfb_4 = RFB_2D(2048, channel)
        else:
            self.rfb_2 = RFB_3D(512, channel)
            self.rfb_3 = RFB_3D(1024, channel)
            self.rfb_4 = RFB_3D(2048, channel)

        self.pa = PartialDecoder(32, dimension)

        ## TODO: change inputs according to variable inputs

        self.reverse_attention1 = ReverseAttention(dimension, 2048, 256, 0.25, 32)
        self.reverse_attention2 = ReverseAttention(dimension, 1024, 64, 2, 16)
        self.reverse_attention3 = ReverseAttention(dimension, 512, 64, 2, 8)
        self.conv_block = ConvolutionBlock(
            1,
            64,
            dimension,
            kernel_size=3,
            stride=2,
            padding=1,
            use_activation=False,
            use_batch_norm=False,
        )

    def forward(self, x):
        # encoder Res2Net

        if self.rgb:
            x = self.resnet.conv1(x)
        else:
            x = self.conv_block(x)

        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)

        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x2_rfb = self.rfb_2(x2)
        x3_rfb = self.rfb_3(x3)
        x4_rfb = self.rfb_4(x4)

        pa = self.pa(x4_rfb, x3_rfb, x2_rfb)
        map_5 = F.interpolate(pa, scale_factor=8, mode=self.mode)

        # why does reverse_attention 1 has 3 conv layer and the others have 2?
        x, map_4 = self.reverse_attention1(pa, x4)
        x, map_3 = self.reverse_attention2(x, x3)
        x, map_2 = self.reverse_attention3(x, x2)

        # return [map_1,map_2,map_3]
        return [map_5, map_4, map_3, map_2]


# x = torch.ones((1,1,256,256))
#
#
# net = PraNet(32,2)
#
# for i in net(x):
#     print(i.shape)

# from torchsummary import summary
# summary(net,input_size=(3,352,352))
