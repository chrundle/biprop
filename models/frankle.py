"""
Replications of models from Frankle et al. Lottery Ticket Hypothesis
"""

import torch.nn as nn
from utils.builder import get_builder

from args import args
import pdb

# Binary activation function with gradient estimator
import torch
class F_BinAct(torch.autograd.Function):
  @staticmethod
  def forward(ctx, inp):
    # Save input for backward
    ctx.save_for_backward(inp)
    # Unscaled sign function
    return torch.sign(inp)

  @staticmethod
  def backward(ctx, grad_out):
    # Get input from saved ctx
    inp, = ctx.saved_tensors
    # Clone grad_out
    grad_input = grad_out.clone()
    # Gradient approximation from quadratic spline
    inp = torch.clamp(inp, min=-1.0, max=1.0)
    inp = 2*(1 - torch.abs(inp))
    # Return gradient
    return grad_input * inp

class BiRealAct(nn.Module):
  def __init__(self):
    super(BiRealAct, self).__init__()

  def forward(self, input):
    return F_BinAct.apply(input)


class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(64 * 16 * 16, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 64 * 16 * 16, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv2_BinAct(nn.Module):
    def __init__(self):
        super(Conv2_BinAct, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            builder.batchnorm(64),
            BiRealAct(),
            builder.conv3x3(64, 64),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(64),
            BiRealAct(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(64 * 16 * 16, 256),
            builder.batchnorm(256),
            BiRealAct(),
            builder.conv1x1(256, 256),
            builder.batchnorm(256),
            BiRealAct(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 64 * 16 * 16, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv4(nn.Module):
    def __init__(self):
        super(Conv4, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(32 * 32 * 8, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 8192, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv4_BinAct(nn.Module):
    def __init__(self):
        super(Conv4_BinAct, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            builder.batchnorm(64),
            BiRealAct(),
            builder.conv3x3(64, 64),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(64),
            BiRealAct(),
            builder.conv3x3(64, 128),
            builder.batchnorm(128),
            BiRealAct(),
            builder.conv3x3(128, 128),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(128),
            BiRealAct(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(32 * 32 * 8, 256),
            builder.batchnorm(256),
            BiRealAct(),
            builder.conv1x1(256, 256),
            builder.batchnorm(256),
            BiRealAct(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 8192, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv6(nn.Module):
    def __init__(self):
        super(Conv6, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(128, 256),
            nn.ReLU(),
            builder.conv3x3(256, 256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(256 * 4 * 4, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 256 * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv6_BinAct(nn.Module):
    def __init__(self):
        super(Conv6_BinAct, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            builder.batchnorm(64),
            BiRealAct(),
            builder.conv3x3(64, 64),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(64),
            BiRealAct(),
            builder.conv3x3(64, 128),
            builder.batchnorm(128),
            BiRealAct(),
            builder.conv3x3(128, 128),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(128),
            BiRealAct(),
            builder.conv3x3(128, 256),
            builder.batchnorm(256),
            BiRealAct(),
            builder.conv3x3(256, 256),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(256),
            BiRealAct(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(256 * 4 * 4, 256),
            builder.batchnorm(256),
            BiRealAct(),
            builder.conv1x1(256, 256),
            builder.batchnorm(256),
            BiRealAct(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 256 * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv6_BNN(nn.Module):
    def __init__(self):
        super(Conv6_BNN, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 128, first_layer=True),
            builder.batchnorm(128),
            BinAct(),
            builder.conv3x3(128, 128),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(128),
            BinAct(),
            builder.conv3x3(128, 256),
            builder.batchnorm(256),
            BinAct(),
            builder.conv3x3(256, 256),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(256),
            BinAct(),
            builder.conv3x3(256, 512),
            builder.batchnorm(512),
            BinAct(),
            builder.conv3x3(512, 512),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(512),
            BinAct()
        )

        self.linear = nn.Sequential(
            builder.conv1x1(512 * 4 * 4, 1024),
            builder.batchnorm(1024),
            BinAct(),
            builder.conv1x1(1024, 1024),
            builder.batchnorm(1024),
            BinAct(),
            builder.conv1x1(1024, 10)
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 512 * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv8(nn.Module):
    def __init__(self):
        super(Conv8, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(128, 256),
            nn.ReLU(),
            builder.conv3x3(256, 256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(256, 512),
            nn.ReLU(),
            builder.conv3x3(512, 512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(512 * 2 * 2, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 512 * 2 * 2, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv8_BinAct(nn.Module):
    def __init__(self):
        super(Conv8_BinAct, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            builder.batchnorm(64),
            BiRealAct(),
            builder.conv3x3(64, 64),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(64),
            BiRealAct(),
            builder.conv3x3(64, 128),
            builder.batchnorm(128),
            BiRealAct(),
            builder.conv3x3(128, 128),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(128),
            BiRealAct(),
            builder.conv3x3(128, 256),
            builder.batchnorm(256),
            BiRealAct(),
            builder.conv3x3(256, 256),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(256),
            BiRealAct(),
            builder.conv3x3(256, 512),
            builder.batchnorm(512),
            BiRealAct(),
            builder.conv3x3(512, 512),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(512),
            BiRealAct(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(512 * 2 * 2, 256),
            builder.batchnorm(256),
            BiRealAct(),
            builder.conv1x1(256, 256),
            builder.batchnorm(256),
            BiRealAct(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 512 * 2 * 2, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        builder = get_builder()
        self.linear = nn.Sequential(
            builder.conv1x1(28 * 28, 300, first_layer=True),
            nn.ReLU(),
            builder.conv1x1(300, 100),
            nn.ReLU(),
            builder.conv1x1(100, 10),
        )

    def forward(self, x):
        out = x.view(x.size(0), 28 * 28, 1, 1)
        out = self.linear(out)
        return out.squeeze()

def scale(n):
    return int(n * args.width_mult)


class Conv4Wide(nn.Module):
    def __init__(self):
        super(Conv4Wide, self).__init__()
        builder = get_builder()

        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            nn.ReLU(),
            builder.conv3x3(scale(64), scale(64)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(64), scale(128)),
            nn.ReLU(),
            builder.conv3x3(scale(128), scale(128)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(128)*8*8, scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(128)*8*8, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv4Wide_BinAct(nn.Module):
    def __init__(self):
        super(Conv4Wide_BinAct, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            builder.batchnorm(scale(64)),
            BiRealAct(),
            builder.conv3x3(scale(64), scale(64)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(64)),
            BiRealAct(),
            builder.conv3x3(scale(64), scale(128)),
            builder.batchnorm(scale(128)),
            BiRealAct(),
            builder.conv3x3(scale(128), scale(128)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(128)),
            BiRealAct(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(128)*8*8, scale(256)),
            builder.batchnorm(scale(256)),
            BiRealAct(),
            builder.conv1x1(scale(256), scale(256)),
            builder.batchnorm(scale(256)),
            BiRealAct(),
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(128)*8*8, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv6Wide(nn.Module):
    def __init__(self):
        super(Conv6Wide, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            nn.ReLU(),
            builder.conv3x3(scale(64), scale(64)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(64), scale(128)),
            nn.ReLU(),
            builder.conv3x3(scale(128), scale(128)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(128), scale(256)),
            nn.ReLU(),
            builder.conv3x3(scale(256), scale(256)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(256) * 4 * 4, scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(256) * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv6Wide_BinAct(nn.Module):
    def __init__(self):
        super(Conv6Wide_BinAct, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            builder.batchnorm(scale(64)),
            BiRealAct(),
            builder.conv3x3(scale(64), scale(64)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(64)),
            BiRealAct(),
            builder.conv3x3(scale(64), scale(128)),
            builder.batchnorm(scale(128)),
            BiRealAct(),
            builder.conv3x3(scale(128), scale(128)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(128)),
            BiRealAct(),
            builder.conv3x3(scale(128), scale(256)),
            builder.batchnorm(scale(256)),
            BiRealAct(),
            builder.conv3x3(scale(256), scale(256)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(256)),
            BiRealAct(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(256) * 4 * 4, scale(256)),
            builder.batchnorm(scale(256)),
            BiRealAct(),
            builder.conv1x1(scale(256), scale(256)),
            builder.batchnorm(scale(256)),
            BiRealAct(),
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(256) * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv8Wide(nn.Module):
    def __init__(self):
        super(Conv8Wide, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            nn.ReLU(),
            builder.conv3x3(scale(64), scale(64)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(64), scale(128)),
            nn.ReLU(),
            builder.conv3x3(scale(128), scale(128)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(128), scale(256)),
            nn.ReLU(),
            builder.conv3x3(scale(256), scale(256)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(256), scale(512)),
            nn.ReLU(),
            builder.conv3x3(scale(512), scale(512)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(512) * 2 * 2, scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(512) * 2 * 2, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv8Wide_BinAct(nn.Module):
    def __init__(self):
        super(Conv8Wide_BinAct, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            builder.batchnorm(scale(64)),
            BiRealAct(),
            builder.conv3x3(scale(64), scale(64)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(64)),
            BiRealAct(),
            builder.conv3x3(scale(64), scale(128)),
            builder.batchnorm(scale(128)),
            BiRealAct(),
            builder.conv3x3(scale(128), scale(128)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(128)),
            BiRealAct(),
            builder.conv3x3(scale(128), scale(256)),
            builder.batchnorm(scale(256)),
            BiRealAct(),
            builder.conv3x3(scale(256), scale(256)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(256)),
            BiRealAct(),
            builder.conv3x3(scale(256), scale(512)),
            builder.batchnorm(scale(512)),
            BiRealAct(),
            builder.conv3x3(scale(512), scale(512)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(512)),
            BiRealAct(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(512) * 2 * 2, scale(256)),
            builder.batchnorm(scale(256)),
            BiRealAct(),
            builder.conv1x1(scale(256), scale(256)),
            builder.batchnorm(scale(256)),
            BiRealAct(),
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(512) * 2 * 2, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv8Wide_BinAct_ReLU(nn.Module):
    def __init__(self):
        super(Conv8Wide_BinAct_ReLU, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            #builder.batchnorm(scale(64)),
            BiRealAct(),
            builder.conv3x3(scale(64), scale(64)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(64)),
            nn.ReLU(),
            BiRealAct(),
            builder.conv3x3(scale(64), scale(128)),
            #builder.batchnorm(scale(128)),
            BiRealAct(),
            builder.conv3x3(scale(128), scale(128)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(128)),
            nn.ReLU(),
            BiRealAct(),
            builder.conv3x3(scale(128), scale(256)),
            #builder.batchnorm(scale(256)),
            BiRealAct(),
            builder.conv3x3(scale(256), scale(256)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(256)),
            nn.ReLU(),
            BiRealAct(),
            builder.conv3x3(scale(256), scale(512)),
            #builder.batchnorm(scale(512)),
            BiRealAct(),
            builder.conv3x3(scale(512), scale(512)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(512)),
            nn.ReLU(),
            #BiRealAct(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(512) * 2 * 2, scale(256)),
            #builder.batchnorm(scale(256)),
            BiRealAct(),
            builder.conv1x1(scale(256), scale(256)),
            builder.batchnorm(scale(256)),
            nn.ReLU(),
            #BiRealAct(),
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(512) * 2 * 2, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv8Wide_BinAct_ReLU_Final_Act(nn.Module):
    def __init__(self):
        super(Conv8Wide_BinAct_ReLU_Final_Act, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            builder.batchnorm(scale(64)),
            BiRealAct(),
            builder.conv3x3(scale(64), scale(64)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(64)),
            #nn.ReLU(),
            BiRealAct(),
            builder.conv3x3(scale(64), scale(128)),
            builder.batchnorm(scale(128)),
            BiRealAct(),
            builder.conv3x3(scale(128), scale(128)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(128)),
            #nn.ReLU(),
            BiRealAct(),
            builder.conv3x3(scale(128), scale(256)),
            builder.batchnorm(scale(256)),
            BiRealAct(),
            builder.conv3x3(scale(256), scale(256)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(256)),
            #nn.ReLU(),
            BiRealAct(),
            builder.conv3x3(scale(256), scale(512)),
            builder.batchnorm(scale(512)),
            BiRealAct(),
            builder.conv3x3(scale(512), scale(512)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(512)),
            #nn.ReLU(),
            BiRealAct(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(512) * 2 * 2, scale(256)),
            builder.batchnorm(scale(256)),
            BiRealAct(),
            builder.conv1x1(scale(256), scale(256)),
            builder.batchnorm(scale(256)),
            nn.ReLU(),
            #BiRealAct(),
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(512) * 2 * 2, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class VGG_Small(nn.Module):
    def __init__(self):
        super(VGG_Small, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 128, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(128),
            #nn.ReLU(),
            builder.conv3x3(128, 256),
            #nn.ReLU(),
            builder.conv3x3(256, 256),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(256),
            #nn.ReLU(),
            builder.conv3x3(256, 512),
            #nn.ReLU(),
            builder.conv3x3(512, 512),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(512),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(512 * 4 * 4, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 512 * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class VGG_Small_BinAct(nn.Module):
    def __init__(self):
        super(VGG_Small_BinAct, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 128, first_layer=True),
            #builder.batchnorm(128),
            BiRealAct(),
            builder.conv3x3(128, 128),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(128),
            #nn.ReLU(),
            BiRealAct(),
            builder.conv3x3(128, 256),
            #builder.batchnorm(256),
            BiRealAct(),
            builder.conv3x3(256, 256),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(256),
            #nn.ReLU(),
            BiRealAct(),
            builder.conv3x3(256, 512),
            #builder.batchnorm(512),
            BiRealAct(),
            builder.conv3x3(512, 512),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(512),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(512 * 4 * 4, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 512 * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class VGG_Small_bn_BinAct(nn.Module):
    def __init__(self):
        super(VGG_Small_bn_BinAct, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 128, first_layer=True),
            builder.batchnorm(128),
            BiRealAct(),
            builder.conv3x3(128, 128),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(128),
            #nn.ReLU(),
            BiRealAct(),
            builder.conv3x3(128, 256),
            builder.batchnorm(256),
            BiRealAct(),
            builder.conv3x3(256, 256),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(256),
            #nn.ReLU(),
            BiRealAct(),
            builder.conv3x3(256, 512),
            builder.batchnorm(512),
            BiRealAct(),
            builder.conv3x3(512, 512),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(512),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(512 * 4 * 4, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 512 * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class VGG_Small_noReLU_BinAct(nn.Module):
    def __init__(self):
        super(VGG_Small_noReLU_BinAct, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 128, first_layer=True),
            #builder.batchnorm(128),
            BiRealAct(),
            builder.conv3x3(128, 128),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(128),
            #nn.ReLU(),
            BiRealAct(),
            builder.conv3x3(128, 256),
            #builder.batchnorm(256),
            BiRealAct(),
            builder.conv3x3(256, 256),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(256),
            #nn.ReLU(),
            BiRealAct(),
            builder.conv3x3(256, 512),
            #builder.batchnorm(512),
            BiRealAct(),
            builder.conv3x3(512, 512),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(512),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(512 * 4 * 4, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 512 * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Wide_VGG_Small(nn.Module):
    def __init__(self):
        super(Wide_VGG_Small, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(16), first_layer=True),
            builder.batchnorm(scale(16)),
            BiRealAct(),
            builder.conv3x3(scale(16), scale(16)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(16)),
            BiRealAct(),
            builder.conv3x3(scale(16), scale(32)),
            builder.batchnorm(scale(32)),
            BiRealAct(),
            builder.conv3x3(scale(32), scale(32)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(32)),
            BiRealAct(),
            builder.conv3x3(scale(32), scale(64)),
            builder.batchnorm(scale(64)),
            BiRealAct(),
            builder.conv3x3(scale(64), scale(64)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(64)),
            #BiRealAct(),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(64) * 4 * 4, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(64) * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Wide_VGG_Small_132(nn.Module):
    def __init__(self):
        super(Wide_VGG_Small_132, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(16), first_layer=True),
            builder.batchnorm(scale(16)),
            #BiRealAct(),
            nn.ReLU(),
            builder.conv3x3(scale(16), scale(16)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(16)),
            #BiRealAct(),
            nn.ReLU(),
            builder.conv3x3(scale(16), scale(32)),
            builder.batchnorm(scale(32)),
            #BiRealAct(),
            nn.ReLU(),
            builder.conv3x3(scale(32), scale(32)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(32)),
            #BiRealAct(),
            nn.ReLU(),
            builder.conv3x3(scale(32), scale(64)),
            builder.batchnorm(scale(64)),
            #BiRealAct(),
            nn.ReLU(),
            builder.conv3x3(scale(64), scale(64)),
            nn.MaxPool2d((2, 2)),
            builder.batchnorm(scale(64)),
            #BiRealAct(),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(64) * 4 * 4, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(64) * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()
