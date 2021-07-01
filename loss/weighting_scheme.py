from copy import deepcopy

import numpy as np
import torch


class WeightScheme:
    def __init__(
        self,
        scheme,
        weights,
        loss_list=None,
        max_epochs=100,
        num_losses=3,
        beta_adapt=0.5,
        num_history=3,
    ):
        self.scheme = scheme
        self.max_epochs = max_epochs

        self.weights = deepcopy(weights)
        self.num_losses = len(weights.keys())
        self.internal_count = 1
        self.loss_list = loss_list
        self.loss_history = torch.zeros((self.num_losses, num_history), device="cuda")


        self.beta_adapt = beta_adapt

        self.num_history = num_history
        self.adapt_ = torch.zeros(
            (self.num_losses,), device="cuda:0"
        )
        self.iter = 0


        self.alphas = torch.zeros(
            (self.num_losses,), requires_grad=False, device="cuda"
        ).type(torch.FloatTensor)
        self.mean_l = torch.zeros(
            (self.num_losses,), requires_grad=False, device="cuda"
        ).type(torch.FloatTensor)
        self.mean_L = torch.zeros(
            (self.num_losses,), requires_grad=False, device="cuda"
        ).type(torch.FloatTensor)
        self.S_l = torch.zeros(
            (self.num_losses,), requires_grad=False, device="cuda"
        ).type(torch.FloatTensor)
        self.std = None

    def __call__(self, *args, **kwargs):
        if self.scheme == "adapt":
            return self.__adapt__(*args, **kwargs)
        if self.scheme == "adapt_weighted":
            return self.__adapt_weighted__(*args, **kwargs)
        if self.scheme == "cov":
            return self.__cov__(*args, **kwargs)

    def __adapt__(self, *args, **kwargs):
        loss_ = kwargs["new_losses"]
        epoch_ = kwargs["epoch"]
        loss_ = loss_.clone().detach()

        if epoch_ >= self.num_history:
            self.loss_history = torch.roll(self.loss_history, 1, 0)
            self.loss_history[:, -1] = loss_
            weights_ = torch.exp(
                self.beta_adapt
                * (
                    self.loss_history[:, self.num_history - 1]
                    - self.loss_history[:, self.num_history - 2]
                )
            )
            self.adapt_ = weights_ / torch.sum(weights_)
            count = 0
            for k, v in self.weights.items():
                self.weights[k] = self.adapt_[count]
                count += 1
            return self.weights
        else:
            # return weightning only based on the prev slopes or adapts
            #self.loss_history[:, epoch_] = loss_
            if epoch_ == 0:
                weights_ = torch.exp(self.beta_adapt * (self.loss_history[:, epoch_]))
                self.adapt_ = weights_ / torch.sum(weights_)
                count= 0
                for k, v in self.weights.items():
                    self.weights[k] = self.adapt_[count]
                    count += 1
                return self.weights

            else:
                weights_ = torch.exp(
                    self.beta_adapt
                    * (self.loss_history[:, epoch_] - self.loss_history[:, epoch_ - 1])
                )
                self.adapt_ = weights_ / torch.sum(weights_)
                count = 0
                for k, v in self.weights.items():
                    self.weights[k] = self.adapt_[count]
                    count += 1
                return self.weights

    def __adapt_weighted__(self, *args, **kwargs):
        loss_ = kwargs["new_losses"]
        epoch_ = kwargs["epoch"]
        loss_ = loss_.detach()

        if epoch_ >= self.num_history:
            self.loss_history = torch.roll(self.loss_history, 1, 0)
            self.loss_history[:, -1] = loss_
            weights_ = loss_ * torch.exp(
                self.beta_adapt
                * (
                    self.loss_history[:, self.num_history - 1]
                    - self.loss_history[:, self.num_history - 2]
                )
            )
            self.adapt_ = weights_ / torch.sum(weights_)
            count = 0
            for k, v in self.weights.items():
                self.weights[k] = self.adapt_[count]
                count += 1
            return self.weights
        else:
            # return weightning only based on the prev slopes or adapts
            self.loss_history[:, epoch_] = loss_
            if epoch_ == 0:
                weights_ = loss_ * torch.exp(
                    self.beta_adapt * (self.loss_history[:, epoch_])
                )
                self.adapt_ = weights_ / torch.sum(weights_)
                count= 0
                for k, v in self.weights.items():
                    self.weights[k] = self.adapt_[count]
                    count += 1
                return self.weights
            else:
                weights_ = loss_ * torch.exp(
                    self.beta_adapt
                    * (self.loss_history[:, epoch_] - self.loss_history[:, epoch_ - 1])
                )
                self.adapt_ = weights_ / torch.sum(weights_)
                count= 0
                for k, v in self.weights.items():
                    self.weights[k] = self.adapt_[count]
                    count += 1
                return self.weights

    def __cov__(self, *args, **kwargs):
        loss_ = kwargs["new_losses"]
        epoch_ = kwargs["epoch"]
        loss_ = loss_.clone().detach()
        l_0 = loss_ if self.iter == 0 else self.mean_L

        # Loss ratio
        l_r = loss_ / l_0

        if self.iter <= 1:
            self.alphas = (
                torch.ones((self.num_losses,), requires_grad=False, device="cuda").type(
                    torch.FloatTensor
                )
                / self.num_losses
            )
        else:
            ls = self.std / self.mean_l
            self.alphas = ls / torch.sum(ls)

        if self.iter == 0:
            mean = 0.0
        else:
            mean = 1.0 - 1 / (self.iter + 1)

        l_u = l_r.clone().detach()

        # Welford Algorithm
        mean_n = mean * self.mean_l + (1 - mean) * l_u
        self.S_l += (l_u - self.mean_l) * (l_u - mean_n)
        self.mean_l = mean_n

        var = self.S_l / (self.iter + 1)
        self.std = torch.sqrt(var + 1e-8)

        self.mean_L = mean * self.mean_L + (1 - mean) * loss_.clone()
        self.iter += 1
        count = 0
        for k,v in self.weights.items():
            self.weights[k] = self.alphas[count]
            count += 1
        return self.weights
