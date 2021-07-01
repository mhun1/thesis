import gym
import torch
from rl.agent import AC
from torch import nn

from rl.memory import Memory


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        n_latent_var,
        lr,
        betas,
        gamma,
        K_epochs,
        eps_clip,
        device="cpu",
    ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = AC(state_dim, action_dim, n_latent_var).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = AC(state_dim, action_dim, n_latent_var).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.obs).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprops).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, path):
        torch.save(self.ppo.policy.state_dict(), "./PPO_{}.pth".format(path))


# mem = Memory()
# env = gym.make('gym_tree:learning_env-v0')
# action_dim = env.action_space.n
# obs_dim = env.observation_space.shape[0]
# ppo = PPO(obs_dim, action_dim, 64, 0.002, (0.9,0.999), 0.99, 4, 0.2)


# Running policy_old:
# epochs = 100
# for i in range(0,epochs):
#     state = get_components(x,y)
#     action = ppo.policy_old(state, mem)
#     decoded_act = decode_action()
#     reward = 1-loss
#     done = False
#     if reward > 0.97:
#         done = True
#
#     # Saving reward and is_terminal:
#     mem.rewards.append(reward)
#     mem.is_terminals.append(done)
#
#     # update if its time
#     if timestep % update_timestep == 0:
#         ppo.update(memory)
#         mem.clear_memory()
#         timestep = 0
#
#     running_reward += reward
#     if render:
#         env.render()
#     if done:
#         break
#
#     avg_length += t
