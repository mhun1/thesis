
import pytorch_lightning as pl
import torch
import torchio as tio
import wandb
from torch.autograd import Variable
from torch.utils.data import DataLoader
from typing import Any, List
from loss.weighting_scheme import WeightScheme
from models.utils.torch_utils import get_function
from summary import get_lr_scheduler
from utils.metrics import get_metrics
from utils.wab_vis import render_volume
from .eval_metrics import avg_val, generate_metrics
from .vis_utils import visualize_batch, draw_tree
from models.mixup import mixup_data, mixup_criterion
import re
from switchcase import switch


class NetWrapper(pl.LightningModule):
    def __init__(
        self,
        model,
        loss,
        train_data,
        val_data,
        batch=4,
        init_lr=0.0001,
        learning_mode="normal",
        log=True,
        lr_scheduler=None,
        max_epochs=400,
        synthetic_result=None,
        tree=None,
        optim="Adam",
        use_mixup=False,
        loss_norm=None,
        val_range=(0, 20),
        weighting_scheme=None,
        dimension=3,
    ):
        super().__init__()
        self.net = model
        self.loss = loss
        self.loss_2 = loss
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.hparams.learning_rate = init_lr

        self.val_range = val_range
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.batch = batch
        self.use_mixup = use_mixup
        self.learning_mode = learning_mode
        self.logging = log
        self.max_epochs = max_epochs
        self.log_once = True
        self.tree = tree
        self.synthetic_result = synthetic_result
        self.loss_norm = loss_norm
        self.weighting = weighting_scheme
        self.dimension = dimension
        self.metrics = get_metrics()

        if weighting_scheme:
            self.weighting_scheme = WeightScheme(
                weighting_scheme, self.loss.weights
            )

        self.metric_dict = {
            "acc": 0,
            "dice": 0,
            "jac": 0,
            "prec": 0,
            "hd": 5000,
            "sens": 0,
            "f2": 0,
            "spec": 0,
        }

    def training_step(self, batch, batch_idx):
        x, y = batch["data"][tio.DATA], batch["label"][tio.DATA]
        if self.dimension == 2:
            x,y = x.squeeze(1).float(), y.squeeze(1).float()

        y = torch.where(y > 0.0, 1.0, 0.0)
        y_a, y_b, lam = None, None, None
        if self.use_mixup:
            x, y_a, y_b, lam = mixup_data(x, y)
            x, y_a, y_b = map(Variable, (x, y_a, y_b))

        pred = self.net(x)
        loss = 0

        for case in switch(self.learning_mode, comp=re.match):
            if case("normal"):
                return self.normal_training_step(pred, y, y_a, y_b, lam)
            if case("find"):
                return self.__apply_find(pred, y, y_a, y_b, lam)
            else:
                print("Cant find this learning mode")

    def training_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("Train/Avg_Loss", avg_loss, logger=True)

    def validation_step(self, batch, batch_idx):

        x, y = batch["data"][tio.DATA], batch["label"][tio.DATA]
        if self.dimension == 2:
            x, y = x.squeeze(1).permute(1,0,2,3).float(), y.squeeze(1).permute(1,0,2,3).float()

        y = torch.where(y > 0.0, 1.0, 0.0)
        pred = self.net(x)
        loss = 0

        for case in switch(self.learning_mode):
            if case("normal"):
                loss = self.loss(pred, y)
                break

            if case("find"):
                tmp_pred, tmp_y = Prediction(torch.sigmoid(pred)), Label(y)
                loss = self.loss(tmp_pred, tmp_y)
                loss = loss.val
                if self.loss_norm:
                    loss = (loss - self.loss_norm[0]) / (
                        self.loss_norm[1] - self.loss_norm[0]
                    )
                    loss /= pred.shape[0]
                else:
                    loss = loss.val / pred.shape[0]
                break
            else:
                print("Cant find this learning mode")

        self.log("Val/Loss", loss, logger=True)
        self.log("val_loss", loss)
        metrics = {"val_loss": loss}
        metrics.update(self.__validation(x, y, pred))
        sigmoid_pred = pred.sigmoid()

        if batch_idx == 0:
            metrics["img"] = x.detach().cpu().to(torch.float32)
            metrics["pred"] = pred.detach().cpu().to(torch.float32)
            metrics["softmax"] = sigmoid_pred.detach().cpu().to(torch.float32)
            metrics["label"] = y.detach().cpu().to(torch.float32)
            pred_threshold = torch.where(pred < 0.5, 0.0, 1.0)
            metrics["threshold"] = pred_threshold.cpu().to(torch.float32)
        return metrics

    def __validation(self, x, y, pred):
        x_cpu, y_cpu = x.cpu(), y.cpu()
        return generate_metrics(x_cpu, y_cpu, pred)

    def validation_epoch_end(self, outputs: List[Any]) -> None:

        new_dict = {}
        new_dict["Accuracy"] = avg_val(outputs,"accuracy")
        new_dict["Dice"] = avg_val(outputs,"dice")
        new_dict["F2_Score"] = avg_val(outputs,"f2_score")
        new_dict["Precision"] = avg_val(outputs,"precision")
        new_dict["Recall"] = avg_val(outputs,"recall")
        new_dict["Jaccard"] = avg_val(outputs,"jaccard")
        new_dict["Hausdorff"] = avg_val(outputs, "hd")

        if self.logging:
            self.__log_image(
                Input=visualize_batch(outputs[0]["img"], range=self.val_range),
                Pred=visualize_batch(outputs[0]["pred"], range=self.val_range),
                Softmax=visualize_batch(outputs[0]["softmax"], range=self.val_range),
                Label=visualize_batch(outputs[0]["label"], range=self.val_range),
            )

            self.__render_volume(
                outputs[0]["img"], outputs[0]["label"], outputs[0]["threshold"]
            )

        if self.learning_mode == "find" and self.log_once:
            if self.tree:
                self.__log_image(Tree=draw_tree(self.tree))
            if self.synthetic_result:
                self.__log_image(Syn_Result=self.synthetic_result)
            self.log_once = False

        if self.logger:
            self.__log_other_dict(new_dict)

        self.metric_dict["acc"] = (
            new_dict["Accuracy"] if new_dict["Accuracy"] > self.metric_dict["acc"] else self.metric_dict["acc"]
        )
        self.metric_dict["dice"] = (
            new_dict["Dice"] if new_dict["Dice"] > self.metric_dict["dice"] else self.metric_dict["dice"]
        )
        self.metric_dict["jac"] = (
            new_dict["Jaccard"] if new_dict["Jaccard"] > self.metric_dict["jac"] else self.metric_dict["jac"]
        )
        self.metric_dict["prec"] = (
            new_dict["Precision"]
            if new_dict["Precision"] > self.metric_dict["prec"]
            else self.metric_dict["prec"]
        )
        self.metric_dict["hd"] = (
            new_dict["Hausdorff"] if (new_dict["Hausdorff"] < self.metric_dict["hd"]) and new_dict["Hausdorff"] != -1 else self.metric_dict["hd"]
        )
        self.metric_dict["sens"] = (
            new_dict["Recall"]
            if new_dict["Recall"] > self.metric_dict["sens"]
            else self.metric_dict["sens"]
        )
        self.metric_dict["f2"] = (
            new_dict["F2_Score"] if new_dict["F2_Score"]  > self.metric_dict["f2"] else self.metric_dict["f2"]
        )

    def on_train_end(self) -> None:
        for name in self.metric_dict.keys():
            self.logger.experiment.log(
                {"best_score/" + name: self.metric_dict[name], "custom_step": 0},
                commit=False,
            )

    def configure_optimizers(self):
        optimizer = get_function(torch.optim, self.optim)(
            self.parameters(), lr=self.hparams.learning_rate
        )

        lr_scheduler = {
            "scheduler": get_lr_scheduler(
                self.lr_scheduler, optimizer, self.max_epochs
            ),
            "name": "PolyLR",
        }

        if optimizer and lr_scheduler:
            return [optimizer], [lr_scheduler]
        elif optimizer and not lr_scheduler:
            return [optimizer]
        else:
            raise ValueError("Optimizer or Lr_Scheduler -> None")

    def train_dataloader(self):

        if self.dimension == 2:
            sampler = tio.data.UniformSampler((1, 224, 224))
            patches_queue = tio.Queue(
                self.train_dataset,
                128,
                12,
                sampler,
                num_workers=8,
            )
            return DataLoader(patches_queue, batch_size=32, pin_memory=True)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):

        if self.val_dataset:
            return DataLoader(
                self.val_dataset, shuffle=False, num_workers=1, pin_memory=True
            )

    def normal_training_step(self, pred, y, y_a, y_b, lam):
        if self.use_mixup:
            loss = mixup_criterion(self.loss, pred, y_a, y_b, lam)
        else:
            loss = self.loss(pred, y)


        if self.weighting == "cov" or self.weighting == "adapt":
            # get single loss tensor
            single_losses = self.loss.fake_forward(pred,y)
            weights = self.weighting_scheme(new_losses=single_losses, epoch=self.current_epoch)
            self.loss.weights = weights
            loss = single_losses[1]*self.loss.weights["alpha"] + single_losses[0]*self.loss.weights["beta"]
            loss = loss.to("cuda")
            self.logger.experiment.log(self.loss.weights,commit=False)

        batch_dict = {"loss": loss}
        return batch_dict



    def __log_image(self, **kwargs):
        for name in kwargs:
            self.logger.experiment.log(
                {name: [wandb.Image(kwargs[name], caption=name)]}, commit=False
            )

    def __log_metrics(self, **kwargs):
        for name in kwargs:
            self.log(name, kwargs[name])

    def __log__weight(self, **kwargs):
        for name in kwargs:
            self.logger.experiment.log({"weights/" + name: kwargs[name]}, commit=False)

    def __render_volume(self, data, label, pred):
        path = render_volume(data, label, pred, dimension=self.dimension)
        wandb.log(
            {"3D Visualisation of Prediction": wandb.Video(path, fps=12, format="gif")}
        )

    def __log_dict(self, dict, prefix="rl_w/"):
        for name in dict.keys():
            self.logger.experiment.log({prefix + name: dict[name]}, commit=False)

    def __log_other_dict(self,dict):
        for name in dict.keys():
            self.log(name,dict[name])

    def __apply_find(self, pred, y, y_a, y_b, lam):
        tmp_pred, y = Prediction(torch.sigmoid(pred)), Label(y)
        if self.use_mixup:
            y_a, y_b = Label(y_a), Label(y_b)
            loss = mixup_criterion(self.loss, tmp_pred, y_a, y_b, lam)
        else:
            loss = self.loss(tmp_pred, y)
            loss = loss.val
            if self.loss_norm:
                loss = (loss - self.loss_norm[0]) / (
                    self.loss_norm[1] - self.loss_norm[0]
                )
                loss /= pred.shape[0]
            else:
                loss = loss.val / pred.shape[0]
        batch_dict = {"loss": loss}
        return batch_dict
