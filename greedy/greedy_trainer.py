import itertools
import os
import warnings
from copy import deepcopy

import numpy as np
import torch.nn as nn
import torch
import higher
import torchio as tio
import wandb
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from models.eval_metrics import generate_metrics_greedy
from summary import get_loss
from utils.args_parser import get_parser
from utils.utils import (
    get_id,
    create_id,
    save_id,
    create_dirs,
    pytorch_logger,
)



def train(model, loss_func, train_dataset, action_dict, every_x_batch=25, model_path=None, epochs=100, max_num=0, check_every_x=2):

    train_loader = DataLoader(
        train_dataset["train"],
        batch_size=2,
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

    loss_func.apply_non_lin = False

    scaler = GradScaler()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    best_score = {"best_score/acc": 0.0, "best_score/dice": 0.0,
                  "best_score/f2": 0.0, "best_score/jac": 0.0,
                  "best_score/prec": 0.0, "best_score/sens": 0.0, "best_score/hd":1000.0}

    for epoch in range(epochs):
        # data loader
        print("---------Epoch {}------------".format(epoch))
        avg_loss = 0
        count = 0
        random_int = np.random.randint(0,max_num)
        update = True if epoch % check_every_x == 0 and epoch else False

        for i, batch in enumerate(train_loader, 0):

            if update and i == random_int:
                model_save = deepcopy(model.state_dict())
                optim_save = deepcopy(opt.state_dict())
                scaler_save = deepcopy(scaler.state_dict())

                min_action = []
                for key, val in action_dict.items():
                    opt.zero_grad()
                    with torch.cuda.amp.autocast():
                        x, y = batch["data"][tio.DATA], batch["label"][tio.DATA]
                        x, y = x.to("cuda"), y
                        y = torch.where(y > 0.0, 1.0, 0.0)
                        pred = model(x)
                        pred = pred.sigmoid().cpu()
                        loss = loss_func.apply_weight(pred, y, val)
                        loss = loss.to("cuda")

                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                    opt.zero_grad()
                    with torch.cuda.amp.autocast():
                        x, y = batch["data"][tio.DATA], batch["label"][tio.DATA]
                        x, y = x.to("cuda"), y
                        y = torch.where(y > 0.0, 1.0, 0.0)
                        pred_2 = model(x)
                        pred_2 = pred_2.sigmoid().cpu()
                        criterion = torch.sum(torch.sqrt((pred_2-y)**2))
                        min_action.append(criterion.item())


                    model.load_state_dict(model_save)
                    opt.load_state_dict(optim_save)
                    scaler.load_state_dict(scaler_save)


                ac = min_action.index(min(min_action))
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    x, y = batch["data"][tio.DATA], batch["label"][tio.DATA]
                    x, y = x.to("cuda"), y
                    y = torch.where(y > 0.0, 1.0, 0.0)
                    pred_updated = model(x)
                    pred_updated = pred_updated.sigmoid().cpu()
                    act = action_dict[ac]
                    loss_func.update(act)
                    loss_updated = loss_func(pred_updated, y)
                    loss_updated = loss_updated.to("cuda")

                scaler.scale(loss_updated).backward()
                scaler.step(opt)
                scaler.update()

                avg_loss += loss_updated.item()
            else:
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    x, y = batch["data"][tio.DATA], batch["label"][tio.DATA]
                    x, y = x.to("cuda").float(), y.to("cuda").float()
                    y = torch.where(y > 0.0, 1.0, 0.0)
                    pred_normal = model(x)
                    pred_normal = pred_normal.sigmoid()
                    loss_normal = loss_func(pred_normal, y)

                scaler.scale(loss_normal).backward()
                scaler.step(opt)
                scaler.update()
                avg_loss += loss_normal.item()
            count += 1

        copy_weights = deepcopy(loss_func.weights)
        copy_weights["epoch"] = epoch
        wandb.log({"Train/Avg_Loss": avg_loss/count, "epoch": epoch})
        wandb.log(copy_weights)
        if epoch % 5 == 0 and epoch != 0:
            model.eval()
            model.float()
            with torch.no_grad():
                num = 0
                avg_val_loss = 0
                out = {
                    "accuracy": 0.0,
                    "dice": 0.0,
                    "jaccard": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "sensitivity": 0.0,
                    "specificity": 0.0,
                    "f2_score": 0.0,
                    "hd": 0.0,
                }

                for i, batch in enumerate(val_loader, 0):
                    x, y = batch["data"][tio.DATA], batch["label"][tio.DATA]
                    x, y = x.to("cuda").float(), y.float()
                    y = torch.where(y > 0.0, 1.0, 0.0)

                    pred = model(x)
                    pred = pred.cpu()
                    pred_2 = pred.clone().sigmoid()
                    val_loss = loss_func(pred_2,y)
                    metrics = generate_metrics_greedy(x, y, pred)
                    out["accuracy"] += metrics["accuracy"]
                    out["dice"] += metrics["dice"]
                    out["jaccard"] += metrics["jaccard"]
                    out["precision"] += metrics["precision"]
                    out["f2_score"] += metrics["f2_score"]
                    out["recall"] += metrics["recall"]
                    out["hd"] += metrics["hd"]
                    avg_val_loss += val_loss
                    num+=1

                wandb.log({"val_loss": avg_val_loss/num, "epoch":epoch})


                total_ = {k: v / num for k, v in out.items()}

                wandb.log({"Hausdorff": total_["hd"], "epoch": epoch})

                new_dict = {}
                new_dict["Accuracy"] = total_["accuracy"]
                new_dict["Dice"] = total_["dice"]
                new_dict["F2_Score"] = total_["f2_score"]
                new_dict["Precision"] = total_["precision"]
                new_dict["Recall"] = total_["recall"]
                new_dict["Jaccard"] = total_["jaccard"]

                best_score["best_score/acc"] = new_dict["Accuracy"] if best_score["best_score/acc"] < new_dict["Accuracy"] else best_score["best_score/acc"]
                best_score["best_score/dice"] = new_dict["Dice"] if best_score["best_score/dice"] < new_dict["Dice"] else best_score["best_score/dice"]
                best_score["best_score/f2"] = new_dict["F2_Score"] if best_score["best_score/f2"] < new_dict["F2_Score"] else best_score["best_score/f2"]
                best_score["best_score/jac"] = new_dict["Jaccard"] if best_score["best_score/jac"] < new_dict["Jaccard"] else best_score["best_score/jac"]
                best_score["best_score/prec"] = new_dict["Precision"] if best_score["best_score/prec"] < new_dict["Precision"] else best_score["best_score/prec"]
                best_score["best_score/sens"] = new_dict["Recall"] if best_score["best_score/sens"] < new_dict["Recall"] else best_score["best_score/sens"]
                best_score["best_score/hd"] = total_["hd"] if best_score["best_score/hd"] > total_["hd"] and total_["hd"] != -1 else best_score["best_score/hd"]

                for k,v in new_dict.items():
                    wandb.log({k:v, "epoch": epoch})
            model.train()

    for k, v in best_score.items():
        wandb.log({k: v, "epoch": epoch})

    if model_path:
        torch.save(model.state_dict(), model_path+"model_.pth")


warnings.filterwarnings("ignore")

if __name__ == "__main__":

    args = get_parser()

    if args.resume == "":
        STORE_PATH, LOG_PATH, CHECKPOINT_PATH = create_dirs(args, args.fold)
    else:
        STORE_PATH = args.resume + "/"
        LOG_PATH = STORE_PATH + "{}" + "/log/"
        CHECKPOINT_PATH = STORE_PATH + "{}" + "/model/"


    args.learning_mode = "g"
    X = np.arange(start=1, stop=51, step=1)
    max_num = 17
    if args.dataset == "Cervical":
        args.epochs = 150
        max_num = 5
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

        act_count = 3
        param_count = len(loss.weights.keys())
        combi = list(itertools.product([i for i in range(3)], repeat=param_count))
        action_dict = {v: k for v, k in enumerate(combi)}
        print(action_dict)

        model = model.cuda()
        print("--------START {} TRAINING--------".format(k))
        train(model, loss, dataset, action_dict, model_path=CHECKPOINT_PATH.format(k), epochs=args.epochs, max_num=max_num)
        wandb.finish()
        k += 1
