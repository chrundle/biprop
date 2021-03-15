"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utils.builder import get_builder
from args import args

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



class BasicBlock_BinAct(nn.Module):
    expansion = 1

    def __init__(self, builder, in_planes, planes, stride=1):
        super(BasicBlock_BinAct, self).__init__()
        self.conv1 = builder.conv3x3(in_planes, planes, stride=stride)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=1)
        self.bn2 = builder.batchnorm(planes)
        self.relu = (lambda: BiRealAct())()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        #out = self.bn2(self.conv2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, builder, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = builder.conv1x1(in_planes, planes)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=stride)
        self.bn2 = builder.batchnorm(planes)
        self.conv3 = builder.conv1x1(planes, self.expansion * planes)
        self.bn3 = builder.batchnorm(self.expansion * planes)
        self.relu = (lambda: BiRealAct())()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class ResNet_BinAct(nn.Module):
    def __init__(self, builder, block, num_blocks):
        super(ResNet_BinAct, self).__init__()
        self.in_planes = 64
        self.builder = builder

        self.conv1 = builder.conv3x3(3, 64, stride=1, first_layer=True)
        self.bn1 = builder.batchnorm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = (lambda: BiRealAct())()

        if args.last_layer_dense:
            self.fc = nn.Conv2d(512 * block.expansion, 10, 1)
        else:
            self.fc = builder.conv1x1(512 * block.expansion, 10)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.builder, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = self.fc(out)
        return out.flatten(1)


class Bottleneck2(nn.Module):
    expansion = 1

    def __init__(self, builder, in_planes, planes, cardinality, stride=1, base_width=64, widen_factor=1):
        super(Bottleneck2, self).__init__()
        width_ratio = planes / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv1 = builder.conv1x1(in_planes, D)
        self.bn1 = builder.batchnorm(D)
        self.conv2 = builder.group_conv3x3(D, D, groups=cardinality, stride=stride)
        self.bn2 = builder.batchnorm(D)
        self.conv3 = builder.conv1x1(D, self.expansion * planes)
        self.bn3 = builder.batchnorm(self.expansion * planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    def __init__(self, in_channels, out_channels, stride, cardinality, widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        D = cardinality * out_channels // widen_factor
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                                                padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


class WideResNeXt_BinAct(nn.Module):
    def __init__(self, builder, block, num_blocks, cardinality, base_width=64, widen_factor=1):
        super(WideResNeXt_BinAct, self).__init__()
        self.in_planes = 64
        self.builder = builder
        self.base_width = base_width
        self.widen_factor = widen_factor

        self.conv1 = builder.conv3x3(3, 64, stride=1, first_layer=True)
        self.bn1 = builder.batchnorm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], cardinality[0], stride=1)
        self.layer2 = self._make_layer(block, 64*(widen_factor+1), num_blocks[1], cardinality[1], stride=2)
        self.layer3 = self._make_layer(block, 128*(widen_factor+1), num_blocks[2], cardinality[2], stride=2)
        self.layer4 = self._make_layer(block, 256*(widen_factor+1), num_blocks[3], cardinality[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

        if args.last_layer_dense:
            self.fc = nn.Conv2d(256*(widen_factor+1) * block.expansion, 10, 1)
        else:
            self.fc = builder.conv1x1(256*(widen_factor+1) * block.expansion, 10)

    def _make_layer(self, block, planes, num_blocks, cardinality, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        #layers = []
        layers = nn.ModuleList()
        for stride in strides:
            layers.append(block(self.builder, self.in_planes, planes, cardinality, stride, self.base_width, self.widen_factor))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = self.fc(out)
        return out.flatten(1)


class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self, cardinality=8, depth=29, widen_factor=4, num_classes=10):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            num_classes: number of classes
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stages[3], num_classes)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        init.kaiming_normal(self.fc.weight)

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv1.forward(x)
        x = F.relu(self.bn1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class WideResNet_BinAct(nn.Module):
    def __init__(self, builder, block, num_blocks, widen_factor=1):
        super(WideResNet_BinAct, self).__init__()
        self.in_planes = 64
        self.builder = builder

        self.conv1 = builder.conv3x3(3, 64, stride=1, first_layer=True)
        self.bn1 = builder.batchnorm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64*(widen_factor+1), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128*(widen_factor+1), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256*(widen_factor+1), num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = (lambda: BiRealAct())()

        if args.last_layer_dense:
            self.fc = nn.Conv2d(256*(widen_factor+1) * block.expansion, 10, 1)
        else:
            self.fc = builder.conv1x1(256*(widen_factor+1) * block.expansion, 10)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.builder, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        #out = self.conv1(x)
        #out = self.(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = self.fc(out)
        return out.flatten(1)


class SmallResNet_BinAct(nn.Module):
    def __init__(self, builder, block, num_blocks):
        super(SmallResNet_BinAct, self).__init__()
        self.in_planes = 16
        self.builder = builder

        self.conv1 = builder.conv3x3(3, 16, stride=1, first_layer=True)
        self.bn1 = builder.batchnorm(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = (lambda: BiRealAct())()

        if args.last_layer_dense:
            self.fc = nn.Conv2d(64 * block.expansion, 10, 1)
        else:
            self.fc = builder.conv1x1(64 * block.expansion, 10)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.builder, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = self.fc(out)
        return out.flatten(1)


def resnext29_8x64d_c10():
    model = CifarResNeXt(cardinality=8, depth=29, widen_factor=4, num_classes=10)
    return model

def cResNet18_BinAct_v2():
    return ResNet_BinAct(get_builder(), BasicBlock_BinAct, [2, 2, 2, 2])

def cWideResNet18_2_BinAct_v2():
    return WideResNet_BinAct(get_builder(), BasicBlock_BinAct, [2, 2, 2, 2], widen_factor=2)

def cWideResNet18_3_BinAct():
    return WideResNet_BinAct(get_builder(), BasicBlock_BinAct, [2, 2, 2, 2], widen_factor=3)

def cResNet34_BinAct():
    return ResNet_BinAct(get_builder(), BasicBlock_BinAct, [3, 4, 6, 3])

#ResNeXt
def cWideResNeXt18_2_BinAct():
    return WideResNeXt_BinAct(get_builder(), Bottleneck2, [1, 2, 6, 2], [4,8,8,16], widen_factor=2)

def cWideResNeXt18_2_BinAct_small():
    return WideResNeXt_BinAct(get_builder(), Bottleneck2, [1, 2, 6, 2], [4,4,8,8], widen_factor=2)


#def cResNet50():
#    return ResNet(get_builder(), Bottleneck, [3, 4, 6, 3])
#
#
#def cResNet101():
#    return ResNet(get_builder(), Bottleneck, [3, 4, 23, 3])
#
#
#def cResNet152():
#    return ResNet(get_builder(), Bottleneck, [3, 8, 36, 3])


def cResNet20_BinAct():
    return SmallResNet_BinAct(get_builder(), BasicBlock_BinAct, [3, 3, 3])

def cResNet32_BinAct():
    return SmallResNet_BinAct(get_builder(), BasicBlock_BinAct, [5, 5, 5])

def cResNet44_BinAct():
    return SmallResNet_BinAct(get_builder(), BasicBlock_BinAct, [7, 7, 7])

def cResNet56_BinAct():
    return SmallResNet_BinAct(get_builder(), BasicBlock_BinAct, [9, 9, 9])

def cResNet110_BinAct():
    return SmallResNet_BinAct(get_builder(), BasicBlock_BinAct, [18, 18, 18])
