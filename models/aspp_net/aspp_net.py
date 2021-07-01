import torch
from torch import nn

from models.backbone.resnet import load_resnet
from models.basic_blocks.aspp import ASPP
from models.basic_blocks.conv_block import ConvolutionBlock


class StrangeDecoder(nn.Module):
    def __init__(self, dimension=3):
        super(StrangeDecoder, self).__init__()
        self.conv3 = ConvolutionBlock(320, 32, dimension=dimension, kernel_size=3)
        self.conv3_1 = ConvolutionBlock(32, 32, dimension=dimension, kernel_size=3)
        self.conv4 = ConvolutionBlock(1281, 256, dimension=dimension, kernel_size=3)
        self.conv4_1 = ConvolutionBlock(256, 256, dimension=dimension, kernel_size=3)
        self.conv5 = ConvolutionBlock(3584, 1024, dimension=dimension, kernel_size=3)
        self.conv5_1 = ConvolutionBlock(1024, 1024, dimension=dimension, kernel_size=3)

        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

    def forward(self, x, skip):
        # target 32,64,128,128
        # skip[0] #64, 32, 64, 64
        # skip[1] #256, 16, 32, 32
        # skip[2] #512, 8, 16, 16
        # skip[3] #1024, 8, 16, 16
        # skip[4] #2048, 8, 16, 16
        # x #1,16,32,32

        cat = torch.cat((skip[4], skip[3], skip[2]), dim=1)
        cat = self.conv5(cat)
        cat = self.conv5_1(cat)
        cat = self.up(cat)

        cat = torch.cat((cat, skip[1], x), dim=1)
        cat = self.conv4(cat)
        cat = self.conv4_1(cat)
        cat = self.up(cat)

        cat = torch.cat((cat, skip[0]), dim=1)
        cat = self.conv3(cat)
        cat = self.conv3_1(cat)
        cat = self.up(cat)
        return cat


class ASPPNet(nn.Module):
    def __init__(self):
        super(ASPPNet, self).__init__()

        self.backbone = load_resnet("/content/backbone/resnet_50.pth")
        # print("require?: ", self.backbone.requires_grad)
        # self.backbone.requires_grad_(False)
        self.aspp_decoder = ASPP(2048, 64, 3)
        self.in_conv = ConvolutionBlock(1, 64, 3)
        self.out_conv = ConvolutionBlock(
            32, 1, 3, use_activation=False, use_batch_norm=False
        )
        # self.out_conv = ConvolutionBlock(32,1,3)

        self.strange_decoder = StrangeDecoder(dimension=3)

    def forward(self, x):
        # last = self.in_conv(x)
        x, skip = self.backbone(x)
        x = self.strange_decoder(x, skip)
        x = self.out_conv(x)
        return x


# x = torch.ones((1,1,64,128,128))
# net = ASPPNet()
