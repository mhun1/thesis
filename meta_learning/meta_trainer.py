import warnings
import numpy as np
import torch
import higher
import torchio as tio

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from loss.region.dice import Dice
from meta_learning.meta_model import MetaLoss, MetaLoss3D, MetaLoss2D
from utils.args_parser import get_parser
from utils.metrics import get_metrics
from utils.utils import create_components



def train(model, meta_model, objective, train_dataset, device="cpu"):
    epochs = 100

    sampler = tio.data.UniformSampler((1, 224, 224))  #
    patches_queue = tio.Queue(
        train_dataset["train"],
        4,
        4,
        sampler,
        num_workers=4,
    )
    train_loader = DataLoader(patches_queue, batch_size=2, pin_memory=False)

    val_loader = DataLoader(
        train_dataset["val"], batch_size=1, shuffle=False, num_workers=1, pin_memory=False
    )

    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)

    sgd = torch.optim.Adam(model.parameters(), lr=0.001)

    metrics = get_metrics()
    for epoch in range(epochs):
        # data loader
        print("---------Epoch {}------------".format(epoch))
        num = 0
        for batch in train_loader:
            meta_optimizer.zero_grad()
            with higher.innerloop_ctx(model, sgd, copy_initial_weights=False) as (
                net,
                diff,
            ):

                x, y = batch["data"][tio.DATA].to(device), batch["label"][tio.DATA].to(device)
                x, y = x.squeeze(1).float(), y.squeeze(1).float()
                y = torch.where(y > 0.0, torch.Tensor([1.0]), torch.Tensor([0.0]))

                pred = net(x)
                pred = pred.sigmoid()

                meta_loss = meta_model(pred,y)
                diff.step(meta_loss)
                task_loss = objective(pred, y)

                task_loss.backward()

                num += 1
            meta_optimizer.step()

        if epoch % 5 == 0:
            net.eval()

            for i, batch in enumerate(val_loader, 0):
                x, y = batch["data"][tio.DATA].to(device), batch["label"][tio.DATA].to(device)
                x, y = x.squeeze(1).float().permute(1,0,2,3), y.squeeze(1).float().permute(1,0,2,3)

                y = torch.where(y > 0.0, torch.Tensor([1.0]), torch.Tensor([0.0]))

                pred = model(x)
                pred = torch.where(pred < 0.5, torch.Tensor([0.0]), torch.Tensor([1.0]))
                pred = pred
                metrics(pred,y.long())
            final_metrics = metrics.compute()
            net.train()



warnings.filterwarnings("ignore")

if __name__ == "__main__":

    args = get_parser()
    args.dimension = 2
    args.dataset = "Cervical"
    args.loss = "TverskyFocal"
    X = np.arange(start=1, stop=51, step=1)
    if args.dataset == "Cervical":
        X = np.arange(start=0, stop=15, step=1)

    kf = KFold(n_splits=args.fold, shuffle=True, random_state=21)

    k = 0
    for train_index, test_index in kf.split(X):

        model, dataset, _, _ = create_components(args, "", train_index, test_index, k, "")
        target_loss = Dice(apply_nonlin=False)

        meta_loss = MetaLoss()

        model = model.cpu()

        meta_loss = meta_loss.cpu()

        print("--------START {} TRAINING--------".format(k))
        train(model, meta_loss, target_loss, dataset)

        k += 1
