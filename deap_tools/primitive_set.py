import torch

from deap import gp

from deap_tools.operations import (
    add,
    minus,
    mul,
    DimCompatible,
    Scalar,
    sum,
    div,
    exp,
    log,
    absol,
    torch_pow,
    torch_sin,
    torch_cos,
    torch_tan,
    torch_sqrt,
    torch_min,
    torch_max,
    torch_mean,
    torch_median,
    torch_std,
    torch_var,
    Prediction,
    Label,
    normal_min,
    normal_max,
)
from loss.distance.active_contour import ActiveContour
from loss.region.dice import Dice
from loss.distribution.focal import Focal


def generate_pset():
    pset = gp.PrimitiveSetTyped("MAIN", [DimCompatible, DimCompatible], Scalar)

    pset.addPrimitive(
        add, [DimCompatible, DimCompatible], DimCompatible, name="plus1", weight=0.6
    )
    pset.addPrimitive(
        add, [Scalar, DimCompatible], DimCompatible, name="plus2", weight=0.2
    )

    pset.addPrimitive(
        minus, [DimCompatible, DimCompatible], DimCompatible, name="minus1", weight=0.6
    )
    pset.addPrimitive(
        minus, [Scalar, DimCompatible], DimCompatible, name="minus2", weight=0.2
    )
    pset.addPrimitive(
        minus, [DimCompatible, Scalar], DimCompatible, name="minus3", weight=0.2
    )

    pset.addPrimitive(
        mul, [DimCompatible, DimCompatible], DimCompatible, name="mul1", weight=0.6
    )
    pset.addPrimitive(
        mul, [Scalar, DimCompatible], DimCompatible, name="mul2", weight=0.2
    )

    pset.addPrimitive(
        div, [DimCompatible, DimCompatible], DimCompatible, name="div1", weight=0.6
    )
    pset.addPrimitive(
        div, [Scalar, DimCompatible], DimCompatible, name="div2", weight=0.2
    )
    pset.addPrimitive(
        div, [DimCompatible, Scalar], DimCompatible, name="div3", weight=0.2
    )

    # pset.addPrimitive(sum, [DimCompatible], DimCompatible, name="sum", weight=0.3)
    pset.addPrimitive(sum, [DimCompatible], Scalar, name="sum", weight=0.7)

    pset.addPrimitive(exp, [DimCompatible], DimCompatible, name="exp", weight=0.3)
    pset.addPrimitive(log, [DimCompatible], DimCompatible, name="log", weight=0.3)

    # pset.addPrimitive(absol, [DimCompatible], DimCompatible, name="abs", weight=0.3)
    # pset.addPrimitive(torch_pow, [DimCompatible,Scalar], DimCompatible, name="pow", weight=0.3)
    # pset.addPrimitive(torch_sin, [DimCompatible], DimCompatible, name="sin", weight=0.3)
    # pset.addPrimitive(torch_sinh, [DimCompatible], DimCompatible, name="sinh", weight=0.3)
    # pset.addPrimitive(torch_cos, [DimCompatible], DimCompatible, name="cos", weight=0.3)
    # pset.addPrimitive(torch_cosh, [DimCompatible], DimCompatible, name="cosh", weight=0.3)
    # pset.addPrimitive(torch_tan, [DimCompatible], DimCompatible, name="tan", weight=0.3)
    # pset.addPrimitive(torch_tanh, [DimCompatible], DimCompatible, name="tanh", weight=0.3)
    # pset.addPrimitive(torch_sqrt, [DimCompatible], DimCompatible, name="sqrt", weight=0.3)

    pset.addPrimitive(torch_min, [DimCompatible], Scalar, name="min", weight=0.2)
    pset.addPrimitive(torch_max, [DimCompatible], Scalar, name="max", weight=0.2)
    # pset.addPrimitive(torch_dist, [DimCompatible, DimCompatible], float, name="dist", weight=0.2)
    pset.addPrimitive(torch_mean, [DimCompatible], Scalar, name="mean", weight=0.5)
    pset.addPrimitive(torch_median, [DimCompatible], Scalar, name="median", weight=0.5)
    pset.addPrimitive(torch_std, [DimCompatible], Scalar, name="std", weight=0.2)
    pset.addPrimitive(torch_var, [DimCompatible], Scalar, name="var", weight=0.2)

    def _(f):
        return f

    pset.addPrimitive(_, [Scalar], Scalar)
    pset.addTerminal(Scalar(torch.Tensor([2])), Scalar, weight=0.2, name="two")
    pset.addTerminal(Scalar(torch.Tensor([1])), Scalar, weight=0.2, name="one")
    pset.addTerminal(Scalar(torch.Tensor([-1])), Scalar, weight=0.2, name="mone")
    pset.addTerminal(Scalar(torch.Tensor([-2])), Scalar, weight=0.2, name="mtwo")
    pset.renameArguments(ARG0="x", weight=0.8)
    pset.renameArguments(ARG1="y", weight=0.8)
    terminal_types = [DimCompatible, Scalar]

    return pset, terminal_types


def generate_pset_losses():
    pset = gp.PrimitiveSetTyped("MAIN", [Prediction, Label], Scalar)

    pset.addPrimitive(add, [Scalar, Scalar], Scalar, name="plus", weight=0.6)
    pset.addPrimitive(minus, [Scalar, Scalar], Scalar, name="minus", weight=0.6)
    pset.addPrimitive(mul, [Scalar, Scalar], Scalar, name="mul", weight=0.6)
    pset.addPrimitive(div, [Scalar, Scalar], DimCompatible, name="div", weight=0.6)

    ## Add functions
    list_of_losses = [
        Dice(evolution=True),
        Focal(evolution=True),
        ActiveContour(evolution=True),
    ]
    for loss in list_of_losses:
        name = type(loss).__name__
        pset.addPrimitive(loss, [Prediction, Label], Scalar, name=name, weight=0.6)

    pset.addPrimitive(exp, [Scalar], Scalar, name="exp", weight=0.5)
    pset.addPrimitive(log, [Scalar], Scalar, name="log", weight=0.5)
    pset.addPrimitive(absol, [Scalar], Scalar, name="abs", weight=0.2)
    pset.addPrimitive(torch_pow, [Scalar, Scalar], Scalar, name="pow", weight=0.45)
    pset.addPrimitive(torch_sin, [Scalar], Scalar, name="sin", weight=0.45)
    pset.addPrimitive(torch_cos, [Scalar], Scalar, name="cos", weight=0.45)
    pset.addPrimitive(torch_tan, [Scalar], Scalar, name="tan", weight=0.45)
    pset.addPrimitive(torch_sqrt, [Scalar], Scalar, name="sqrt", weight=0.45)
    pset.addPrimitive(normal_min, [Scalar, Scalar], Scalar, name="min", weight=0.2)
    pset.addPrimitive(normal_max, [Scalar, Scalar], Scalar, name="max", weight=0.2)
    pset.addPrimitive(torch_mean, [Scalar], Scalar, name="mean", weight=0.2)
    pset.addPrimitive(torch_median, [DimCompatible], Scalar, name="median", weight=0.2)
    pset.addPrimitive(torch_std, [DimCompatible], Scalar, name="std", weight=0.2)
    pset.addPrimitive(torch_var, [DimCompatible], Scalar, name="var", weight=0.2)

    def _(f):
        return f

    pset.addPrimitive(_, [Scalar], Scalar, weight=0.0001)
    pset.addPrimitive(_, [Prediction], Prediction, weight=0.0001)
    pset.addPrimitive(_, [Label], Label, weight=0.0001)

    pset.addTerminal(Scalar(torch.Tensor([2])), Scalar, weight=0.2, name="two")
    pset.addTerminal(Scalar(torch.Tensor([1])), Scalar, weight=0.2, name="one")
    pset.addTerminal(Scalar(torch.Tensor([-1])), Scalar, weight=0.2, name="mone")
    pset.addTerminal(Scalar(torch.Tensor([-2])), Scalar, weight=0.2, name="mtwo")
    pset.renameArguments(ARG0="x", weight=0.8)
    pset.renameArguments(ARG1="y", weight=0.8)
    terminal_types = [Prediction, Label, Scalar]
    return pset, terminal_types
