import os
import warnings
import numpy as np
import wandb
from sklearn.model_selection import KFold

from models.lightning_wrapper import NetWrapper
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from summary import get_loss

from utils.args_parser import get_parser
from utils.utils import (
    create_dirs,
    create_components,
    create_id,
    get_id,
    save_id,
)

warnings.filterwarnings("ignore")
args = get_parser()


if __name__ == "__main__":

    if args.resume == "":
        STORE_PATH, LOG_PATH, CHECKPOINT_PATH = create_dirs(args, args.fold)
    else:
        STORE_PATH = args.resume + "/"
        LOG_PATH = STORE_PATH + "{}" + "/log/"
        CHECKPOINT_PATH = STORE_PATH + "{}" + "/model/"

    sample_count = len([name for name in os.listdir(args.dir)])
    X = np.arange(start=0, stop=sample_count, step=1)
    kf = KFold(n_splits=args.fold, shuffle=True, random_state=21)
    k = 0
    metric_list = []

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

        checkpoint = ModelCheckpoint(
            dirpath=CHECKPOINT_PATH.format(k),
            filename=PREFIX + "-{Dice:.4f}",
            mode="max",
            monitor="Dice",
            # save_last=True,
        )

        model, dataset, logger, lr_monitor = create_components(
            args, LOG_PATH.format(k), train_index, test_index, k, run_id["run_id"]
        )


        trainer = pl.Trainer(
            max_epochs=args.epochs,
            logger=logger,
            gpus=1,
            checkpoint_callback=checkpoint,
            check_val_every_n_epoch=args.check_val,
            num_sanity_val_steps=0,
            callbacks=[lr_monitor],
            precision=16,
            profiler="simple",
        )

        if args.dimension == 2:
            trainer = pl.Trainer(
                max_epochs=args.epochs,
                logger=logger,
                gpus=1,
                checkpoint_callback=checkpoint,
                check_val_every_n_epoch=args.check_val,
                num_sanity_val_steps=0,
                callbacks=[lr_monitor],
            )

        net_wrapper = NetWrapper(
            model,
            get_loss(args.loss),
            dataset["train"],
            dataset["val"],
            batch=args.batch,
            init_lr=args.initial_lr,
            lr_scheduler=args.lr,
            max_epochs=args.epochs,
            optim=args.optimizer,
            use_mixup=args.mixup,
            weighting_scheme=args.weighting,
            learning_mode=args.learning_mode,
            dimension=args.dimension,
        )

        trainer.fit(net_wrapper)
        metric_list.append(net_wrapper.metric_dict)

        wandb.finish()
        k += 1


