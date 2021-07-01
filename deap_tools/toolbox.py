import os

import pytorch_lightning as pl
import numpy as np

import wandb

from deap import base, creator, gp, tools
from deap.gp import PrimitiveTree


from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold

from deap_tools.primitive_set import generate_pset_losses, generate_pset
from evolution.synthetic_data import synthetic_data_test, synthetic_load_data
from models.lightning_wrapper import NetWrapper

from utils.utils import create_components, create_id
import torchio as tio
import shutil


class EvoTool:
    def __init__(self, pset, args):

        X = np.arange(start=1, stop=51, step=1)
        y = np.arange(start=1, stop=51, step=1)
        kf = KFold(n_splits=args.fold, shuffle=True, random_state=21)

        k = 0
        split = kf.split(X)
        train_index, val_index = next(split)

        self.args = args

        self.pset, self.terminal_types = generate_pset()
        # self.pset = generate_pset_losses()

        self.base_epoch = 5
        self.args.loss = "EvoLoss"
        self.model, self.dataset, _, _, self.dim = create_components(
            self.args, "", train_index, val_index, 0, create_id()
        )

        self.path_to_store = "/home/mhun/store/evolution/"
        self.current_indiv = 0
        self.current_gen = 0
        self.max_gen = args.evo_gen
        self.max_pop = args.population
        self.init_toolbox()

    def init_toolbox(self):
        self.toolbox = base.Toolbox()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create(
            "Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset
        )

        def condition(height, depth):
            """Expression generation stops when the depth is equal to height."""
            return depth == height

        self.toolbox.register(
            "expr",
            gp.generate_unsafe,
            pset=self.pset,
            min_=2,
            max_=4,
            condition=condition,
            types=self.terminal_types,
        )

        # self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.toolbox.register(
            "evaluate",
            self.weight,
            model=self.model,
            dataset=self.dataset,
            toolbox=self.toolbox,
        )

        self.toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.2)
        self.toolbox.register(
            "mutate",
            gp.mutNodeReplacement,
            pset=self.pset,  # , individual=creator.Individual, pset=self.pset
        )
        self.toolbox.register("select", tools.selRoulette)

    def weight(self, individual, model, dataset, toolbox):
        loss = 0
        tree = PrimitiveTree(individual)
        score = self.inspect_tree(tree)

        if score == 0:
            print(
                "{}|{}|{}|SCORE: {}".format(
                    self.current_gen, self.current_indiv, tree, loss
                )
            )
            self.current_indiv += 1
            if self.current_indiv > self.max_pop:
                self.current_gen += 1
                self.current_indiv = 0
            return (loss,)

        func = toolbox.compile(expr=individual)

        loss += 1
        label = self.dataset["train"][0]["label"][tio.DATA].squeeze(0).float()

        log_path = (
            self.path_to_store
            + str(self.current_gen)
            + "_"
            + str(self.current_indiv)
            + "/"
        )
        os.mkdir(log_path)

        # score,out,fig,min_val,max_val = synthetic_data_test(label,func)
        score, out, fig, min_val, max_val = synthetic_load_data(func)

        if (
            not out["str_dec"]
            or not out["non_inc"]
            or max(score) > 100
            or min(score) < -100
            or np.isnan(score).any()
            or np.isinf(score).any()
        ):

            print(
                "{}|{}|{}|SCORE: {}".format(
                    self.current_gen, self.current_indiv, tree, loss
                )
            )
            self.current_indiv += 1
            if self.current_indiv > self.max_pop:
                self.current_gen += 1
                self.current_indiv = 0
            shutil.rmtree(log_path)
            return (loss,)

        id = create_id()
        logger = WandbLogger(
            name="{}|{}|{}".format(self.current_gen, self.current_indiv, "placeholder"),
            save_dir=log_path,
            project="Evolution" + self.args.dataset,
            id=id["run_id"],
            group="Evo+" + str(self.current_gen),
        )

        trainer = pl.Trainer(
            max_epochs=self.base_epoch + self.current_gen,
            limit_train_batches=0.25,
            logger=logger,
            gpus=1,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            precision=16,
            weights_summary=None,
            progress_bar_refresh_rate=50,
        )

        net_wrapper = NetWrapper(
            model,
            func,
            dataset["train"],
            dataset["val"],
            self.dim,
            batch=self.args.batch,
            init_lr=self.args.initial_lr,
            lr_scheduler=self.args.lr_scheduler,
            optim=self.args.optimizer,
            use_mixup=self.args.mixup,
            learning_mode="find",
            synthetic_result=fig,
            tree=tree,
            loss_norm=(min_val, max_val),
        )

        trainer.fit(net_wrapper)
        wandb.finish()
        loss += float(np.mean(net_wrapper.score)) * 3

        del net_wrapper

        print(
            "{}|{}|{}|SCORE: {}".format(
                self.current_gen, self.current_indiv, tree, loss
            )
        )
        self.current_indiv += 1
        if self.current_indiv > self.max_pop:
            self.current_gen += 1
            self.current_indiv = 0

        return (loss,)

    def inspect_tree(self, tree):

        # if x and y are not contained -> 0
        # if tree is too small < 3 -> minus score
        # if tree is too big > 27 -> minus score

        terminal_dict = {"ARG0": 0, "ARG1": 0, "ARG2": 0, "ARG3": 0}

        for i in tree:
            if i.name in terminal_dict.keys():
                terminal_dict[i.name] += 1

        if terminal_dict["ARG0"] == 0 or terminal_dict["ARG1"] == 0:
            return 0

        return 1
