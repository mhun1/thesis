import re


import torch.nn as nn
import torch
from datasets.lumbar import dataset_lumbar
from loss.compound.cel1 import BCEL1
from lr.poly_lr import get_poly_lr
from models.unet.unet import UNet
from datasets.transforms import get_transforms, get_transforms_lumbar
from datasets.cervical import dataset_cervical
from switchcase import switch

def get_dataset(name, train_list, val_list, dir):
    out = {"train": None, "val": None}
    if name == "Cervical":
        t_tr, v_tr = get_transforms(shape=32, other=True)
        out["train"] = dataset_cervical(train_list, t_tr, path=dir)
        out["val"] = dataset_cervical(val_list, v_tr, path=dir)
    elif name == "Lumbar":
        t_tr, v_tr = get_transforms_lumbar(shape=32)
        out["train"] = dataset_lumbar(train_list, t_tr, skip_list=[0, 33], path=dir)
        out["val"] = dataset_lumbar(val_list, v_tr, skip_list=[0, 33], path=dir)
    else:
        raise NotImplementedError("This dataset {} does not exist".format(name))
    return out

def get_model(name, dimension=3, depth=5):
    if name == "UNet3D":
        model = UNet(1, 1, dimension, depth=depth, bilinear=False)
    elif name == "UNet2D":
        model = UNet(1, 1, 2, depth=depth, bilinear=False)
    return model

def get_lr_scheduler(name, optim, max_iter):
    if name == "PolyLR":
        return get_poly_lr(optim, max_iter)
    else:
        return None

from loss.compound.bce_dice import BCEDice
from loss.compound.dice_focal import DiceFocal
from loss.compound.tversky_focal import TverskyFocal
from loss.distance.hausdorff import Hausdorff, RebalanceHausdorff, PureHausdorff
from loss.region.effectiveness import Effectiveness
from loss.region.tversky import Tversky
from loss.region.assymetric import Asymmetric
from loss.region.dice import Dice, LogDice
from loss.distribution.focal import Focal
from loss.distance.surface import Surface, RebalanceSurface, PureSurface


def get_loss(name, apply_non_lin=True, device="cuda"):
    for case in switch(name, comp=re.fullmatch):
        if case("Asymmetric"):
            return Asymmetric(beta=1.52, apply_non_lin=apply_non_lin)
        if case("BCE"):
            return nn.BCEWithLogitsLoss(pos_weight=torch.full([1], 5, device=device))
        if case("BCEDice"):
            return BCEDice(apply_non_lin=apply_non_lin)
        if case("L1"):
            return BCEL1()
        if case("Dice"):
            return Dice(apply_nonlin=apply_non_lin)
        if case("LogDice"):
            return LogDice(apply_nonlin=apply_non_lin)
        if case("DiceFocal"):
            return DiceFocal(apply_non_lin=apply_non_lin)
        if case("Effectiveness"):
            return Effectiveness()
        if case("Focal"):
            return Focal()
        if case("Hausdorff"):
            return Hausdorff()
        if case("PureHausdorff"):
            return PureHausdorff()
        if case("PureSurface"):
            return PureSurface()
        if case("Surface"):
            return Surface()
        if case("RebalanceHausdorff"):
            return RebalanceHausdorff()
        if case("RebalanceSurface"):
            return RebalanceSurface()
        if case("Tversky"):
            return Tversky(apply_non_lin=apply_non_lin)
        if case("TverskyFocal"):
            return TverskyFocal(apply_non_lin=apply_non_lin)
