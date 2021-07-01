import numpy as np
import torch
from deap import gp


class DimCompatible:
    def __init__(self, val, repr=None):
        self.val = val.cuda()
        self.repr = repr

    def __repr__(self):
        if self.repr:
            return self.repr


class Prediction(DimCompatible):
    def __init__(self, val):
        self.val = val.cuda()


class Label(DimCompatible):
    def __init__(self, val):
        self.val = val.cuda()


class Scalar:
    def __init__(self, val):
        if isinstance(val, torch.Tensor):
            self.val = val.cuda()
        else:
            self.val = torch.Tensor([val]).cuda()


def add(a, b):
    return DimCompatible(a.val + b.val)


def minus(a, b):
    return DimCompatible(a.val + b.val)


def mul(a, b):
    return DimCompatible(a.val * b.val)


def div(a, b):
    return DimCompatible(a.val / b.val)


def sumf(a):
    if isinstance(a, DimCompatible):
        return DimCompatible(torch.sum(a.val).unsqueeze(0))
    else:
        return DimCompatible(a)


def sum(a):
    return Scalar(torch.sum(a.val).unsqueeze(0))


def absol(a):
    return DimCompatible(torch.abs(a.val))


def exp(a):
    return DimCompatible(torch.exp(a.val))


def log(a):
    return DimCompatible(torch.log(a.val))


def torch_pow(a, b):
    return DimCompatible(torch.pow(a.val, b.val))


def torch_sin(a):
    return DimCompatible(torch.sin(a.val))


def torch_sinh(a):
    return DimCompatible(torch.sinh(a.val))


def torch_cos(a):
    return DimCompatible(torch.cos(a.val))


def torch_cosh(a):
    return DimCompatible(torch.cosh(a.val))


def torch_tan(a):
    return DimCompatible(torch.tan(a.val))


def torch_tanh(a):
    return DimCompatible(torch.tanh(a.val))


def torch_sqrt(a):
    if a.val > 0.0:
        return DimCompatible(torch.sqrt(a.val))
    return Scalar(0.0)


def torch_min(a):
    return Scalar(torch.argmin(a.val))


def torch_max(a):
    return Scalar(torch.argmax(a.val))


def torch_dist(a, b):
    return Scalar(torch.dist(a.val, b.val))


def torch_mean(a):
    return Scalar(torch.mean(a.val))


def torch_median(a):

    return Scalar(torch.median(a.val))


def torch_std(a):
    return Scalar(torch.std(a.val))


def torch_var(a):
    return Scalar(torch.var(a.val))


def objective(a, b):
    return a * a + b * b + a * b


def normal_min(a, b):
    return Scalar(min([a.val, b.val]))


def normal_max(a, b):
    return Scalar(max([a.val, b.val]))


def fitness(pred, label):
    return np.linalg.norm(pred - label)


def plot_graph(expr):
    nodes, edges, labels = gp.graph(expr)

    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()
