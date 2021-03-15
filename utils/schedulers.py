import numpy as np

__all__ = ["multistep_lr", "cosine_lr", "constant_lr", "xnor_lr", "get_policy", "mpt11_w18_2_v2_lr", "mpt11_w18_2_lr", "mpt132_w34_2_lr", "mpt11_w34_2_v2_lr","cos_multi_lr"]


def get_policy(name):
    if name is None:
        return constant_lr

    out_dict = {
        "constant_lr": constant_lr,
        "cosine_lr": cosine_lr,
        "multistep_lr": multistep_lr,
        "xnor_lr": xnor_lr,
        "mpt11_w18_2_v2_lr":mpt11_w18_2_v2_lr,
        "mpt11_w18_2_lr":mpt11_w18_2_lr,
        "mpt132_w34_2_lr":mpt132_w34_2_lr,
        "mpt11_w34_2_v2_lr":mpt11_w34_2_v2_lr,
        "cos_multi_lr":cos_multi_lr,
    }

    return out_dict[name]


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def constant_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def cosine_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            e = epoch - args.warmup_length
            es = args.epochs - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def multistep_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    def _lr_adjuster(epoch, iteration):
        lr = args.lr * (args.lr_gamma ** (epoch // args.lr_adjust))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def cos_multi_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 at epoch 30 and 40"""

    def _lr_adjuster(epoch, iteration):
        if epoch >= 30:
            lr = args.lr * 0.1

            if epoch >= 44:
                lr = lr * 0.1

            if epoch >= 54:
                lr = lr * 0.1

            if epoch >= 55:
                e = epoch - args.warmup_length
                es = args.epochs - args.warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * lr
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def xnor_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 at epoch 30 and 40"""

    def _lr_adjuster(epoch, iteration):
        if epoch >= 30: 
            lr = args.lr * 0.1

            if epoch >= 44:
                lr = lr * 0.1

            if epoch >= 54:
                lr = lr * 0.1
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def mpt132_w34_2_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 at epoch 30 and 40"""

    def _lr_adjuster(epoch, iteration):
        if epoch >= 27:
            lr = args.lr * 0.1

            if epoch >= 44:
                lr = lr * 0.1

            if epoch >= 54:
                lr = lr * 0.01
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def mpt11_w34_2_v2_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 at epoch 30 and 40"""

    def _lr_adjuster(epoch, iteration):
        if epoch >= 50:
            lr = args.lr * 0.1

            if epoch >= 62:
                lr = lr * 0.1

            if epoch >= 70:
                lr = lr * 0.01
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def mpt11_w18_2_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 at epoch 30 and 40"""

    def _lr_adjuster(epoch, iteration):
        if epoch >= 46:
            lr = args.lr * 0.1

            if epoch >= 65:
                lr = lr * 0.1

            if epoch >= 80:
                lr = lr * 0.01
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def mpt11_w18_2_v2_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 at epoch 30 and 40"""

    def _lr_adjuster(epoch, iteration):
        if epoch >= 35:
            lr = args.lr * 0.1

            if epoch >= 50:
                lr = lr * 0.1

            if epoch >= 60:
                lr = lr * 0.01
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length
