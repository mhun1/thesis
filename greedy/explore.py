# TODO: DEFINE ACTIONS / RETURN ACTIONS WITH MINIMAL TRAINING LOSS

import torch

from itertools import permutations
import itertools


def get_action(i):
    if i == 0:
        return 0.95
    elif i == 1:
        return 1
    else:
        return 1.05


# Get all permutations of [1, 2, 3]
# perm = permutations([1, 2, 3],3)

# (0) -> -10% | 1: -> 1 | 2: -> +10%
# (0,0,0)
# (0,0,1)
# (0,1,1)
# (0,

# repeat -> count params
# list -> actions
# count: actions^params
# Cartesian product (with one iterable and itself):

#
# act_count = 3
# param_count = 3
# combi = list(itertools.product([i for i in range(3)], repeat=param_count))
# action_dict = {v: k for v, k in enumerate(combi)}
