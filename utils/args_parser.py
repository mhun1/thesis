import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-initial_lr", metavar="L", type=float, default=0.001)
    parser.add_argument("-batch", metavar="BS", type=int, default=4)
    parser.add_argument("-epochs", metavar="E", type=int, default=100)
    parser.add_argument("-check_val", metavar="E", type=int, default=5)
    parser.add_argument("-fold", metavar="E", type=int, default=3)
    parser.add_argument("-evo_gen", metavar="G", type=int, default=40)
    parser.add_argument("-population", metavar="P", type=int, default=30)
    parser.add_argument("-dimension", metavar="D", type=int, default=3)


    parser.add_argument("-show", metavar="E", type=bool, default=False)

    parser.add_argument("-mixup", metavar="MX", type=bool, default=True)
    parser.add_argument("-dataset", metavar="N", type=str)

    # "adapt", "cov", "adapt_weighted"
    parser.add_argument(
        "-weighting", metavar="W", type=str, default=None
    )

    #data dir
    parser.add_argument("-dir", metavar="DN", type=str, default="")
    #model dir
    parser.add_argument("-dir_m", metavar="MD", type=str, default="")

    parser.add_argument("-loss", metavar="N", type=str)
    parser.add_argument("-model", metavar="N", type=str, default="UNet3D")
    parser.add_argument("-optimizer", metavar="O", type=str, default="Adam")
    parser.add_argument("-resume", metavar="RS", type=str, default="")
    parser.add_argument("-lr", metavar="LS", type=str, default="PolyLR")
    parser.add_argument("-learning_mode", metavar="LM", type=str, default="normal")

    args = parser.parse_args()
    print_info(args)
    return args


def print_info(args):
    print(
        "--------------------- \n"
        "MODEL: {} \n"
        "DATSET: {} \n"
        "LOSS: {} \n"
        "EPOCHS: {} \n"
        "BATCH_SIZE: {} \n"
        "LEARNING_RATE: {} \n"
        "LR_SCHEDULER: {} \n"
        "USE_MIXUP: {} \n".format(
            args.model,
            args.dataset,
            args.loss,
            args.epochs,
            args.batch,
            args.initial_lr,
            args.lr,
            args.mixup,
        )
    )
    print("--------------------- \n")
