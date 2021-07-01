import torch
import torch.nn as nn


def add_module(list, module):
    if module:
        list.append(module)


def get_function(module, string):
    if string and module:
        activation_function = getattr(module, string)
        return activation_function if activation_function else None


def sigmoid_help(tensor):
    out = torch.sigmoid(tensor.view(-1)).view(tensor.shape)
    return out


def softmax_help(tensor):
    # softmax = nn.LogSoftmax(dim=0)
    out = torch.softmax(tensor.view(-1)).view(tensor.shape)
    return out


def drop_dimension(input):
    return input.squeeze(0)


def get_up(dimension, scale=2, align_corners=True):
    mode = "bilinear"
    if dimension == 3:
        mode = "trilinear"
    return nn.Upsample(scale_factor=scale, mode=mode, align_corners=align_corners)
