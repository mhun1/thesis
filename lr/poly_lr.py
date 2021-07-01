from torch.optim.lr_scheduler import LambdaLR

def get_poly_lr(optim, max_iter, power=0.9):
    lmbda = lambda epoch: (1 - float(epoch) / max_iter) ** power
    scheduler = LambdaLR(optim, lr_lambda=lmbda)
    return scheduler
