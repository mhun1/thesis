import itertools
import os
import warnings
from copy import deepcopy

import numpy as np
import torch.nn as nn
import torch
import torchio as tio
import wandb
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from loss.region.dice import Dice
from summary import get_loss
from utils.args_parser import get_parser

from utils.utils import (
    get_id,
    create_id,
    save_id,
    create_dirs,
    pytorch_logger,
)
from utils.metrics import get_metrics



def train(model, loss_func, train_dataset):
    epochs = 100
    train_loader = DataLoader(
        train_dataset["train"],
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        train_dataset["val"],
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    scaler = GradScaler()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        print("---------Epoch {}------------".format(epoch))
        for i, batch in enumerate(train_loader, 0):

            opt.zero_grad()
            with torch.cuda.amp.autocast():
                x, y = batch["data"][tio.DATA], batch["label"][tio.DATA]
                x, y = x.to("cuda"), y.to("cuda")

                y = torch.where(y > 0.0, 1.0, 0.0)
                pred = model(x)
                loss = loss_func(pred, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        if epoch % 5 == 0 and epoch != 0:
            model.eval()
            model.float()
            for i, batch in enumerate(val_loader, 0):
                x, y = batch["data"][tio.DATA], batch["label"][tio.DATA]
                x, y = x.to("cuda").float(), y.to("cuda").float()
                y = torch.where(y > 0.0, 1.0, 0.0)
                pred = model(x)
            model.train()



warnings.filterwarnings("ignore")

if __name__ == "__main__":

    args = get_parser()

    if args.resume == "":
        STORE_PATH, LOG_PATH, CHECKPOINT_PATH = create_dirs(args, args.fold)
    else:
        STORE_PATH = args.resume + "/"
        LOG_PATH = STORE_PATH + "{}" + "/log/"
        CHECKPOINT_PATH = STORE_PATH + "{}" + "/model/"

    args.dataset = "Lumbar"
    args.loss = "Tversky"
    args.learning_mode = "g"
    X = np.arange(start=1, stop=51, step=1)
    if args.dataset == "Cervical":
        X = np.arange(start=0, stop=15, step=1)

    kf = KFold(n_splits=args.fold, shuffle=True, random_state=21)
    k = 0
    for train_index, test_index in kf.split(X):
        print("Current fold: {}".format(k))
        print("Train fold: {}".format(train_index))
        print("Val fold: {}".format(test_index))
        print("LOG_PATH: ", LOG_PATH.format(k))
        print("MODEL_PATH: ", CHECKPOINT_PATH.format(k))

        PREFIX = "{}-fold-{}-{}-{}".format(k, args.dataset, args.model, args.loss)
        csv_path = STORE_PATH + "/{}/"

        if args.resume and os.path.isfile(csv_path + "data.csv"):
            run_id = get_id(csv_path.format(k))
        else:
            run_id = create_id()
            save_id(run_id, (csv_path.format(k)))

        model, dataset = pytorch_logger(
            args, LOG_PATH.format(k), train_index, test_index, k, run_id["run_id"]
        )

        loss = get_loss(args.loss)



        model = model.cuda()
        print("--------START {} TRAINING--------".format(k))
        train(model, loss, dataset)
        wandb.finish()
        k += 1
