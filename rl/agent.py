# IDEA:

# GENERAL LOSS / DEPENDENCE OF 3 VALUES

# DISCRETE SET OF ACTIONS / DECREASE / INCREASE

# APPROACH: GREEDY / REINFORCEMENT / EVOLUTION

import torch
from torch import nn
from torch.distributions import Categorical


class AC(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_var, device="cpu"):
        super(AC, self).__init__()
        # actor

        self.device = device
        self.action_layer = nn.Sequential(
            nn.Linear(obs_dim, latent_var),
            nn.Tanh(),
            nn.Linear(latent_var, latent_var),
            nn.Tanh(),
            nn.Linear(latent_var, act_dim),
            nn.Softmax(dim=-1),
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(obs_dim, latent_var),
            nn.Tanh(),
            nn.Linear(latent_var, latent_var),
            nn.Tanh(),
            nn.Linear(latent_var, 1),
        )

    def act(self, state, mem=None):
        state = torch.from_numpy(state).float().to(self.device)
        # print("STATES: ", state)
        action_probs = self.action_layer(state)
        # print(action_probs)
        dist = Categorical(action_probs)
        # print("DIST: ", dist)
        action = dist.sample()
        if mem:
            mem.obs.append(state)
            mem.actions.append(action)
            mem.logprops.append(dist.log_prob(action))
        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
