from copy import deepcopy

import torch
import gym
import numpy as np
import torchio as tio

# Creates once at the beginning of training
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from rl.memory import Memory
from rl.ppo import PPO
from summary import get_loss
from utils.args_parser import get_parser
from utils.get_components import get_components
from utils.utils import create_components


def train(
    model,
    ppo,
    dataset,
    memory,
    env,
    loss_func,
    epochs=200,
    batch_size=1,
    update_timestep=5,
    rl_epochs=5,
):

    train_loader = DataLoader(
        dataset["train"], batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    )
    last_loss = 1
    scaler = torch.cuda.amp.GradScaler()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    timestep = 0
    action = 0
    for epoch in range(epochs):
        model_save = deepcopy(model.state_dict())
        opt_save = deepcopy(opt.state_dict())
        scaler_save = deepcopy(scaler.state_dict())
        for i, batch in enumerate(train_loader, 0):
            opt.zero_grad()
            # Casts operations to mixed precision
            for i in range(rl_epochs):
                with torch.cuda.amp.autocast():
                    x, y = batch["data"][tio.DATA], batch["label"][tio.DATA]
                    y_hat = model(x)
                    tp, tn, fp, fn = get_components(y_hat, y)
                    current_loss = loss_func(y_hat, y)
                    state = np.array([tp, tn, fp, fn, current_loss, action])
                    # sample action from the observation [TP,FP,FN,TN,weights[],current_loss, last_loss]

                    action = ppo.policy_old.act(state, memory)
                    loss_func.apply_weight(action_dict[action])

                scaler.scale(current_loss).backward()
                scaler.step(opt)
                scaler.update()

                with torch.cuda.amp.autocast():
                    x, y = batch["data"][tio.DATA], batch["label"][tio.DATA]
                    y_hat = model(x)
                    last_loss = loss_func(y_hat, y)
                    tp_a, tn_a, fp_a, fn_a = get_components(y_hat, y)
                    reward = tp_a / (tp_a + fp_a) + tn_a / (tn_a + fn_a)
                    done = False
                    if reward > 1.86:
                        done = True

                    memory.rewards.append(reward)
                    memory.terminals.append(done)

                model.load_state_dict(model_save)
                opt.load_state_dict(opt_save)
                scaler.load_state_dict(scaler_save)

            ppo.update(memory)
            memory.clear()

            with torch.cuda.amp.autocast():
                x, y = batch["data"][tio.DATA], batch["label"][tio.DATA]
                y_hat = model(x)
                tp, tn, fp, fn = get_components(y_hat, y)
                current_loss = loss_func(y_hat, y)
                state = np.array([tp, tn, fp, fn, current_loss, action])
                action = ppo.policy_old.act(state, memory)
                loss_func.apply_weight(action_dict[action])


            scaler.update()


if __name__ == "__main__":
    memory = Memory()
    env = gym.make("gym_tree:learning_env-v0")
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    ppo = PPO(obs_dim, action_dim, 64, 0.002, (0.9, 0.999), 0.99, 4, 0.2)
    updated_timestep = 20

    timestep = 3

    args = get_parser()

    X = np.arange(start=1, stop=51, step=1)
    if args.dataset == "Cervical":
        X = np.arange(start=0, stop=15, step=1)

    kf = KFold(n_splits=args.fold, shuffle=True, random_state=21)

    k = 0
    for train_index, test_index in kf.split(X):

        model, dataset, _, _, _ = create_components(
            args, "", train_index, test_index, k, ""
        )

        loss_func = get_loss(args.loss)
        model = model.cuda()

        print("--------START {} TRAINING--------".format(k))
        train(model, ppo, dataset, memory, env, loss_func)
        k += 1

    train()
