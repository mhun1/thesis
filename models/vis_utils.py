import networkx as nx
import torch
import torchvision
import matplotlib.pyplot as plt
import pygraphviz as pgv
import deap
from networkx.drawing.nx_agraph import graphviz_layout


def visualize_batch(batch, range=None, colorbar=False, n_row=5):
    # input shape: z w h

    if isinstance(batch, torch.Tensor):
        if len(batch.shape) > 3:
            batch = batch.squeeze(0).squeeze(0)

    if isinstance(batch, list):
        batch = torch.stack(batch, dim=0)

    grid = None
    if isinstance(range, tuple):
        tst = batch[range[0] : range[1], :, :].unsqueeze(1)
        grid = torchvision.utils.make_grid(tst, nrow=n_row)
    elif not range:
        grid = torchvision.utils.make_grid(batch, nrow=n_row)

    # for grid z _ w h

    fig = plt.figure()
    plt.imshow(grid[0], cmap="gray")
    if colorbar:
        plt.colorbar()
    plt.close(fig)
    return fig


def draw_tree(expr):
    fig = plt.figure()
    nodes, edges, labels = deap.gp.graph(expr)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.close(fig)
    return fig
