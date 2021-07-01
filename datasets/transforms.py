from torchio import ZNormalization, HistogramStandardization, RandomAffine, RandomNoise
from torchio.transforms import Lambda, RescaleIntensity, Compose

import torch
import torch.nn.functional as F
from einops import rearrange


def rearr_standard(x):
    return rearrange(x, "b w h z -> b z h w")


def rearr_other(x):
    return rearrange(x, "b w h z -> b w h z")


def dtype(x):
    if x.dtype != torch.float16:
        return x.to(torch.float16)
    return x


def resize_to_shape_64(x, shape=(64, 224, 224)):
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=shape, mode="trilinear")
    x = x.squeeze(0)
    return x


def resize_to_shape_32(x, shape=(32, 224, 224)):
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=shape, mode="trilinear")
    x = x.squeeze(0)
    return x


def get_transforms(shape=64, other=False):
    func = resize_to_shape_32
    if shape == 64:
        func = resize_to_shape_64

    func_rearr = rearr_standard
    if other:
        func_rearr = rearr_other

    train_transform = Compose(
        {
            RandomAffine(degrees=30): 0.4,
            RandomNoise(): 0.3,
            RescaleIntensity((0, 1), percentiles=(0.5, 99.5)): 1.0,
            ZNormalization(): 1.0,
            Lambda(func_rearr): 1.0,
            Lambda(func): 1.0,
            Lambda(dtype): 1.0,
        }
    )

    val_transform = Compose(
        {
            RescaleIntensity((0, 1), percentiles=(0.5, 99.5)): 1.0,
            ZNormalization(): 1.0,
            Lambda(func_rearr): 1.0,
            Lambda(func): 1.0,
            Lambda(dtype): 1.0,
        }
    )
    return train_transform, val_transform


def get_transforms_lumbar(shape=64, other=False):
    func = resize_to_shape_32
    if shape == 64:
        func = resize_to_shape_64

    func_rearr = rearr_standard
    if other:
        func_rearr = rearr_other

    train_transform = Compose(
        {
            RandomAffine(degrees=30): 0.3,
            RandomNoise(): 0.2,
            RescaleIntensity((0, 1), percentiles=(0.5, 99.5)): 1.0,
            ZNormalization(): 1.0,
            Lambda(func_rearr): 1.0,
            Lambda(func): 1.0,
            Lambda(dtype): 1.0,
        }
    )

    val_transform = Compose(
        {
            RescaleIntensity((0, 1), percentiles=(0.5, 99.5)): 1.0,
            ZNormalization(): 1.0,
            Lambda(func_rearr): 1.0,
            Lambda(func): 1.0,
            Lambda(dtype): 1.0,
        }
    )
    return train_transform, val_transform
