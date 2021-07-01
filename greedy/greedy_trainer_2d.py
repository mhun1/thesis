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

from loss.region.dice import Dice
from loss.region.tversky import Tversky
from meta_learning.meta_model import MetaLoss
from models.eval_metrics import hd, jaccard
from models.vis_utils import visualize_batch
from summary import get_loss
from utils.args_parser import get_parser
from utils.get_components import p_n_blocks
from utils.utils import (
    create_components,
    get_id,
    create_id,
    save_id,
    create_dirs,
    pytorch_logger,
)
from utils.metrics import get_metrics
from utils.wab_vis import render_volume


def train(model, loss_func, train_dataset, action_dict, every_x_batch=25, model_path=None, epochs=100):


    sampler = tio.data.UniformSampler((1, 224, 224))  #
    patches_queue = tio.Queue(
        train_dataset["train"],
        64,
        8,
        sampler,
        num_workers=4,
    )
    train_loader = DataLoader(patches_queue, batch_size=16, pin_memory=True)


    val_loader = DataLoader(
        train_dataset["val"],
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    metrics = get_metrics()
    loss_func.apply_non_lin = False


    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    best_score = {"best_score/acc": 0.0, "best_score/dice": 0.0,
                  "best_score/f2": 0.0, "best_score/jac": 0.0,
                  "best_score/prec": 0.0, "best_score/sens": 0.0, "best_score/hd":1000.0}

    for epoch in range(epochs):
        print("---------Epoch {}------------".format(epoch))
        avg_loss = 0
        count = 0

        for batch in train_loader:
            model_save = deepcopy(model.state_dict())
            optim_save = deepcopy(opt.state_dict())

            min_action = []
            x, y = batch["data"][tio.DATA].squeeze(1).float(), batch["label"][tio.DATA].squeeze(1).float()
            x, y = x.to("cuda"), y.to("cuda")
            y = torch.where(y > 0.0, 1.0, 0.0)
            for key, val in action_dict.items():
                # apply action and forward through model
                opt.zero_grad()
                pred = model(x).sigmoid()
                loss = loss_func.apply_weight(pred, y, val)
                loss.backward()
                opt.step()

                # check criterion on updated model
                pred_2 = model(x).sigmoid()
                criterion = torch.sum(torch.sqrt((pred_2-y)**2))
                min_action.append(criterion.item()/pred_2.shape[0])

                model.load_state_dict(model_save)
                opt.load_state_dict(optim_save)

            ac = min_action.index(min(min_action))
            opt.zero_grad()

            pred_updated = model(x).sigmoid()
            loss_func.update(action_dict[ac])
            loss_updated = loss_func(pred_updated, y)

            loss_updated.backward()
            opt.step()

            avg_loss += loss_updated.item()
            count += 1

        copy_weights = deepcopy(loss_func.weights)
        copy_weights["epoch"] = epoch
        wandb.log({"Train/Avg_Loss": avg_loss/count, "epoch": epoch})
        wandb.log(copy_weights)
        if epoch % 5 == 0 and epoch != 0:
            model.eval()
            with torch.no_grad():
                num = 0
                avg_val_loss = 0
                for i, batch in enumerate(val_loader, 0):
                    x, y = batch["data"][tio.DATA], batch["label"][tio.DATA]
                    x, y = x.squeeze(1).permute(1, 0, 2, 3).float().to("cuda"), y.squeeze(1).permute(1, 0, 2, 3).float()
                    y = torch.where(y > 0.0, 1.0, 0.0)

                    pred = model(x).cpu()
                    pure_pred = pred.clone().cpu()
                    pred = pred.sigmoid()
                    pred_softmax = pred.clone().detach().cpu()
                    val_loss = loss_func(pred,y)
                    pred = torch.where(pred < 0.5, 0.0, 1.0)
                    y = y.long().cpu()
                    metrics(pred, y)
                    avg_val_loss += val_loss
                    num+=1
                    if i == 0:
                        x = x.cpu()
                        wandb.log({"Input": [wandb.Image(visualize_batch(x,n_row=4), caption="Input")]})
                        wandb.log({"Pred": [wandb.Image(visualize_batch(pure_pred, n_row=4), caption="Pred")]})
                        wandb.log({"Softmax": [wandb.Image(visualize_batch(pred_softmax, n_row=4), caption="Softmax")]})
                        wandb.log({"Label": [wandb.Image(visualize_batch(y,n_row=4), caption="Label")]})
                        path = render_volume(x, y, pred, dimension=2)
                        wandb.log(
                            {"3D Visualisation of Prediction": wandb.Video(path, fps=12, format="gif")}
                        )


                total_ = metrics.compute()

                wandb.log({"val_loss": avg_val_loss/num, "epoch":epoch})

                new_dict = {}
                new_dict["Accuracy"] = total_["Accuracy"]
                new_dict["Dice"] = total_["F1"]
                new_dict["F2_Score"] = total_["FBeta"]
                new_dict["Precision"] = total_["Precision"]
                new_dict["Recall"] = total_["Recall"]
                new_dict["Jaccard"] = total_["IoU"][1]
                new_dict["Hausdorff"] = total_["Hausdorff"]

                best_score["best_score/acc"] = new_dict["Accuracy"] if best_score["best_score/acc"] < new_dict["Accuracy"] else best_score["best_score/acc"]
                best_score["best_score/dice"] = new_dict["Dice"] if best_score["best_score/dice"] < new_dict["Dice"] else best_score["best_score/dice"]
                best_score["best_score/f2"] = new_dict["F2_Score"] if best_score["best_score/f2"] < new_dict["F2_Score"] else best_score["best_score/f2"]
                best_score["best_score/jac"] = new_dict["Jaccard"] if best_score["best_score/jac"] < new_dict["Jaccard"] else best_score["best_score/jac"]
                best_score["best_score/prec"] = new_dict["Precision"] if best_score["best_score/prec"] < new_dict["Precision"] else best_score["best_score/prec"]
                best_score["best_score/sens"] = new_dict["Recall"] if best_score["best_score/sens"] < new_dict["Recall"] else best_score["best_score/sens"]
                best_score["best_score/hd"] = new_dict["Hausdorff"] if best_score["best_score/hd"] > new_dict["Hausdorff"] and new_dict["Hausdorff"] != -1 else best_score["best_score/hd"]

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

    args.epochs = 150
    args.dimension = 2
    args.dataset = "Cervical"
    args.loss = "Effectiveness"
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

        act_count = 3
        param_count = len(loss.weights.keys())
        combi = list(itertools.product([i for i in range(3)], repeat=param_count))
        action_dict = {v: k for v, k in enumerate(combi)}
        print(action_dict)

        model = model.cuda()
        print("--------START {} TRAINING--------".format(k))
        train(model, loss, dataset, action_dict, model_path=CHECKPOINT_PATH.format(k), epochs=args.epochs)
        wandb.finish()
        k += 1
