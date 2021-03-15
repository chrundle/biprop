from functools import partial
import os
import pathlib
import shutil
import math

import torch
import torch.nn as nn
from args import args as parser_args


def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "model_best.pth"))

        if not save:
            if filename.exists():
                print("file exists")
                os.remove(filename)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


# Updated to allow for training of batchnorm parameters without learning convolution/dense layer weights
def freeze_model_weights(model):
    print("=> Freezing model weights")
    if parser_args.learn_batchnorm is True: #trains bias, weights, scores
        for n, m in model.named_modules():
            if not hasattr(m, "bias") or m.bias is None:
                if hasattr(m, "weight") and m.weight is not None:
                    print(f"==> No gradient to {n}.weight")
                    m.weight.requires_grad = False
                    if m.weight.grad is not None:
                        print(f"==> Setting gradient of {n}.weight to None")
                        m.weight.grad = None

        # For tuning batchnorm parameters after identifying subnet (will not learn subnet mask)
        if parser_args.tune_batchnorm is True: #trains bias, weights
            for n, m in model.named_modules():
                if hasattr(m, "scores"):
                    m.scores.requires_grad = False
                    print(f"==> No gradient to {n}.scores")
                    if m.scores.grad is not None:
                        print(f"==> Setting gradient of {n}.scores to None")
                        m.scores.grad = None

    elif parser_args.bn_bias_only is True: #trains bias, scores
        for n, m in model.named_modules():
            if hasattr(m, "weight") and m.weight is not None:
                print(f"==> No gradient to {n}.weight")
                m.weight.requires_grad = False
                if m.weight.grad is not None:
                    print(f"==> Setting gradient of {n}.weight to None")
                    m.weight.grad = None
        if parser_args.tune_batchnorm is True: #trains bias only
            for n, m in model.named_modules():
                if hasattr(m, "scores"):
                    m.scores.requires_grad = False
                    print(f"==> No gradient to {n}.scores")
                    if m.scores.grad is not None:
                        print(f"==> Setting gradient of {n}.scores to None")
                        m.scores.grad = None

    else: #trains scores only
        for n, m in model.named_modules():
            if hasattr(m, "weight") and m.weight is not None:
                print(f"==> No gradient to {n}.weight")
                m.weight.requires_grad = False
                if m.weight.grad is not None:
                    print(f"==> Setting gradient of {n}.weight to None")
                    m.weight.grad = None

                if hasattr(m, "bias") and m.bias is not None:
                    print(f"==> No gradient to {n}.bias")
                    m.bias.requires_grad = False

                    if m.bias.grad is not None:
                        print(f"==> Setting gradient of {n}.bias to None")
                        m.bias.grad = None

def get_params(model):
    weights = []
    biases = [0]
    scores = []
    weights_grad = []
    scores_grad = []
    for n,m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print (n," WEIGHTS: ", torch.isnan(m.weight).any())
            weights.extend(m.weight.tolist())
            if m.weight.grad is not None:
                weights_grad.extend(m.weight.grad.tolist())
        if hasattr(m, "bias") and m.bias is not None:
            biases.extend(m.bias.tolist())
        if hasattr(m, "scores") and m.scores is not None:
            print (n," SCORES: ", torch.isnan(m.scores).any())
            scores.extend(m.scores.tolist())
            if m.scores.grad is not None:
                scores_grad.extend(m.scores.grad.tolist())

    return weights,biases, scores, weights_grad, scores_grad

def freeze_model_subnet(model):
    print("=> Freezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            m.scores.requires_grad = False
            print(f"==> No gradient to {n}.scores")
            if m.scores.grad is not None:
                print(f"==> Setting gradient of {n}.scores to None")
                m.scores.grad = None


def unfreeze_model_weights(model):
    print("=> Unfreezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> Gradient to {n}.weight")
            m.weight.requires_grad = True
            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> Gradient to {n}.bias")
                m.bias.requires_grad = True


def unfreeze_model_subnet(model):
    print("=> Unfreezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            print(f"==> Gradient to {n}.scores")
            m.scores.requires_grad = True

def bn_weight_init(model,weight,bias):
    for n,m in model.named_modules():
        if (str(type(m))=="<class 'utils.bn_type.AffineBatchNorm'>"):
            if hasattr(m, "weight") and m.weight is not None:
                if weight is not None:
                    torch.nn.init.uniform_(m.weight,0,(1/(1200**0.5)))
                    print(f"==> Setting weight of {n} to uniform (0,1/sqrt(1200))")
            if hasattr(m, "bias") and m.bias is not None:
                if bias is not None:
                    torch.nn.init.constant_(m.bias, bias)
                    print(f"==> Setting bias of {n} to {bias}")

def set_model_prune_rate(model, prune_rate):
    print(f"==> Setting prune rate of network to {prune_rate}")

    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            m.set_prune_rate(prune_rate)
            print(f"==> Setting prune rate of {n} to {prune_rate}")


def accumulate(model, f):
    acc = 0.0

    for child in model.children():
        acc += accumulate(child, f)

    acc += f(model)

    return acc


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SubnetL1RegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, temperature=1.0):
        l1_accum = 0.0
        for n, p in model.named_parameters():
            if n.endswith("scores"):
                l1_accum += (p*temperature).sigmoid().sum()

        return l1_accum
