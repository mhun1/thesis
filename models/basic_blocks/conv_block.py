from torch import nn
from models.utils.torch_utils import get_function, add_module
import torch


def weight_init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module,nn.Conv3d):
        #nn.init.xavier_uniform_(module.weight, gain=1.0)
        nn.init.trunc_normal_(module.weight,mean=0,std=0.05)
        if module.bias is not None:
            module.bias.data.fill_(0.1)

class ConvolutionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dimension,
        dilation=1,
        kernel_size=3,
        padding=1,
        padding_mode="zeros",
        use_batch_norm=True,
        use_bias=True,
        use_activation=True,
        activation="LeakyReLU",
        #activation="ReLU",
        preactivation=False,
        stride=1,
    ):
        super(ConvolutionBlock, self).__init__()
        building_block = nn.ModuleList()

        self.conv = get_function(nn, "Conv{}d".format(dimension))(
            in_channels,
            out_channels,
            dilation=dilation,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            bias=use_bias,
            stride=stride,
        )

        self.batch_norm = get_function(nn, "InstanceNorm{}d".format(dimension))
        #self.batch_norm = get_function(nn, "BatchNorm{}d".format(dimension))


        self.activiation = None
        if use_activation:
            self.activiation = get_function(nn, activation)()

        if preactivation:
            if use_batch_norm:
                add_module(building_block, self.batch_norm(in_channels))
            add_module(building_block, self.activiation)
            add_module(building_block, self.conv)
        else:
            add_module(building_block, self.conv)
            if use_batch_norm:
                #add_module(building_block, self.batch_norm(out_channels, momentum=0.9) -> 2D
                add_module(building_block, self.batch_norm(out_channels))

            add_module(building_block, self.activiation)

        self.block = nn.Sequential(*building_block)
        #self.block.apply(weight_init)

    def forward(self, x):
        return self.block(x)
