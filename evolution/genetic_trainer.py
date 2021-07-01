##TODO:
import warnings

import numpy as np
from deap import algorithms, tools

from deap_tools.primitive_set import generate_pset
from deap_tools.toolbox import EvoTool
from utils.args_parser import get_parser

import logging

logging.getLogger("lightning").setLevel(0)

warnings.filterwarnings("ignore")
args = get_parser()

pset = generate_pset()
evo_tool = EvoTool(pset, args)


stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)


pop = evo_tool.toolbox.population(n=args.population)
hof = tools.HallOfFame(1)
pop, log = algorithms.eaSimple(
    pop,
    evo_tool.toolbox,
    0.5,
    0.1,
    args.evo_gen,
    stats=mstats,
    halloffame=hof,
    verbose=True,
)
