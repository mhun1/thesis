import csv
import os
import datetime

import wandb
import torch
import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger


from summary import get_dataset, get_model


def write_metric_to_csv(metric_list, filepath):
    from collections import defaultdict

    super_dict = defaultdict(set)  # uses set to avoid duplicates

    for d in metric_list:
        for k, v in d.items():  # use d.iteritems() in python 2
            super_dict[k].add(v)

    for key in super_dict:
        tmp_arr = list(super_dict[key])
        tmp_arr = [i.numpy() if (isinstance(i, torch.Tensor)) else i for i in tmp_arr]
        mean, std = np.mean(tmp_arr) * 100, np.std(tmp_arr) * 100
        super_dict[key] = "{:.3f}({:.3f})".format(mean, std)

    with open(filepath + ".csv", "w") as f:
        w = csv.DictWriter(f, super_dict.keys())
        w.writeheader()
        w.writerow(super_dict)


def create_dirs(args, k):


    custom_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M") + "/"
    MODEL_PATH = args.dir_m
    STORE_PATH = MODEL_PATH + custom_prefix
    LOG_PATH = STORE_PATH + "{}" + "/log/"
    CHECKPOINT_PATH = STORE_PATH + "{}" + "/model/"

    if not os.path.isdir(STORE_PATH):
        os.makedirs(STORE_PATH)
        for i in range(0, k):
            os.makedirs(LOG_PATH.format(i))
            os.makedirs(CHECKPOINT_PATH.format(i))

    return STORE_PATH, LOG_PATH, CHECKPOINT_PATH


def create_components(
    args, log_path, train_list, val_list, k, run_id, project_prefix="Masterthesis"
):
    dataset = get_dataset(args.dataset, train_list, val_list, args.dir)

    model = get_model(args.model, dimension=args.dimension)
    print(
        "WB LOGGER: ",
        str(k) + "-fold+" + args.model + "+" + args.loss + "+" + args.learning_mode,
    )

    weighting = ("+" + args.weighting) if args.weighting else ""
    learning_mode = "" if args.learning_mode == "normal" else ("+" + args.learning_mode)
    print("WEIGHTING: ", weighting)
    if weighting != "":
        learning_mode = weighting

    name = str(k)+ "-fold+" + args.model + "+" + args.loss
    group = args.model + "+" + args.loss + learning_mode

    if args.dimension == 2:
        project_prefix += "2D"

    logger = WandbLogger(
        name=name,
        save_dir=log_path,
        project=project_prefix + args.dataset,
        id=run_id,
        group=group,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    return model, dataset, logger, lr_monitor


def pytorch_logger(
    args, log_path, train_list, val_list, k, run_id, project_prefix="Masterthesis"
):
    dataset = get_dataset(args.dataset, args.remote, train_list, val_list)



    model = get_model(args.model, dimension=args.dimension)
    weighting = ("+" + args.weighting) if args.weighting else ""
    learning_mode = "" if args.learning_mode == "normal" else ("+" +args.learning_mode)
    print("WEIGHTING: ", weighting)
    if weighting != "":
        learning_mode = weighting

    name = str(k)+ "-fold+" + args.model + "+" + args.loss + "+"
    group = args.model + "+" + args.loss + learning_mode

    print(
        "WB LOGGER: ",
        str(k) + "-fold+" + args.model + "+" + args.loss + "+" + args.learning_mode,
    )

    if args.dimension == 2:
        project_prefix += "2D"

    wandb.init(
        project=project_prefix+args.dataset,
        name=name,
        dir=log_path,
        id=run_id,
        group=group,
    )
    return model, dataset

def create_id():
    return {"run_id": wandb.util.generate_id()}

def save_id(dict, path, filename="data"):
    with open(path + filename + ".csv", "w") as f:
        w = csv.DictWriter(f, dict.keys())
        w.writeheader()
        w.writerow(dict)

def get_id(path, filename="data"):
    return next(csv.DictReader(open(path + filename + ".csv")))


