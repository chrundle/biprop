import torch.nn as nn

from utils.builder import get_builder
from args import args


from collections import OrderedDict

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


# BasicBlock {{{
class BasicBlock(nn.Module):
    M = 2
    expansion = 1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(BasicBlock, self).__init__()
        if base_width / 64 > 1:
            raise ValueError("Base width >64 does not work for BasicBlock")

        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn1 = builder.batchnorm(planes)
        self.relu = (lambda: BiRealAct())()
        self.conv2 = builder.conv3x3(planes, planes)
        self.bn2 = builder.batchnorm(planes, last_bn=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# BasicBlock }}}
class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, builder, inplanes, planes,  stride, groups, base_width, widen_factor, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            groups: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = planes / (widen_factor * 64.)
        D = groups * int(base_width * width_ratio)
        self.conv1 = builder.conv1x1(inplanes, D)
        self.bn1 = builder.batchnorm(D)
        self.conv2 = builder.group_conv3x3(D, D, groups=groups)
        self.bn2 = builder.batchnorm(D)
        self.conv3 = builder.conv1x1(D, planes)
        self.bn3 = builder.batchnorm(planes, last_bn=True)
        self.relu = (lambda: BiRealAct())()
        self.downsample = downsample
        self.stride = stride

        self.shortcut = nn.Sequential()
        if inplanes != planes:
            self.shortcut.add_module('shortcut_conv',
                                     builder.conv1x1(inplanes, planes))
            self.shortcut.add_module('shortcut_bn', builder.batchnorm(planes))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

        # bottleneck = self.conv_reduce.forward(x)
        # bottleneck = self.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        # bottleneck = self.conv_conv.forward(bottleneck)
        # bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        # bottleneck = self.conv_expand.forward(bottleneck)
        # bottleneck = self.bn_expand.forward(bottleneck)
        # residual = self.shortcut.forward(x)
        # return F.relu(residual + bottleneck, inplace=True)

class Bottleneck2(nn.Module):
    M = 3
    expansion = 4

    def __init__(self, builder, inplanes, planes, groups, stride=1, downsample=None, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * base_width / 64)
        self.conv1 = builder.conv1x1(inplanes, width)
        self.bn1 = builder.batchnorm(width)
        self.conv2 = builder.conv3x3(width, width, stride=stride)
        self.bn2 = builder.batchnorm(width)
        self.conv3 = builder.conv1x1(width, planes * self.expansion)
        self.bn3 = builder.batchnorm(planes * self.expansion, last_bn=True)
        self.relu = (lambda: BiRealAct())()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

# Bottleneck {{{
class Bottleneck(nn.Module):
    M = 3
    expansion = 4

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * base_width / 64)
        self.conv1 = builder.conv1x1(inplanes, width)
        self.bn1 = builder.batchnorm(width)
        self.conv2 = builder.conv3x3(width, width, stride=stride)
        self.bn2 = builder.batchnorm(width)
        self.conv3 = builder.conv1x1(width, planes * self.expansion)
        self.bn3 = builder.batchnorm(planes * self.expansion, last_bn=True)
        self.relu = (lambda: BiRealAct())()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


# Bottleneck }}}
class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, builder, groups, layers, num_classes=1000, base_width=64, widen_factor=4):
        """ Constructor
        Args:
            groups: number of convolution groups.
            layers: number of layers.
            num_classes: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.groups = groups
        self.layers = layers
        self.block_depth = (self.layers - 2) // 9
        self.base_width = base_width
        if self.base_width // 64 > 1:
            print(f"==> Using {self.base_width // 64}x wide model")

        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = builder.batchnorm(64)
        self.relu = (lambda: BiRealAct())()

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], self.groups[0], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], self.groups[0], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], self.groups[0], 2)
        self.classifier = nn.Linear(self.stages[3], num_classes)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, inplanes, planes, groups, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            inplanes: number of input channels
            planes: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(inplanes, planes, pool_stride, self.groups,
                                                          self.base_width, self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(planes, planes, 1, self.groups, self.base_width,
                                                   self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.classifier(x)


class BasicBlock_C(nn.Module):
    """
    increasing groups is a more effective way of
    gaining accuracy than going deeper or wider
    """

    def __init__(self, builder, inplanes, bottleneck_width=4, groups=32, stride=1, expansion=2):
        super(BasicBlock_C, self).__init__()
        inner_width = groups * bottleneck_width
        width = int(inplanes * bottleneck_width / 64)
        width_ratio = inplanes / (expansion * 64.)
        D = groups * int(bottleneck_width * width_ratio)
        self.expansion = expansion
        self.relu = (lambda: BiRealAct())()
        self.basic = nn.Sequential(OrderedDict(
            [
                ('conv1_0', builder.conv1x1(inplanes, inner_width, stride)),
                ('bn1', builder.batchnorm(inner_width)),
                ('act0', (lambda: BiRealAct())()),
                ('conv3_0', builder.group_conv3x3(inner_width, inner_width, groups=groups, stride=stride)),
                ('bn2', builder.batchnorm(inner_width)),
                ('act1', (lambda: BiRealAct())()),
                ('conv1_1', builder.conv1x1(inner_width, inner_width * self.expansion)),
                ('bn3', builder.batchnorm(inner_width * self.expansion))]))
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != inner_width * self.expansion:
            self.shortcut = nn.Sequential(
                builder.conv1x1(inplanes, inner_width * self.expansion)
            )
        self.bn0 = builder.batchnorm(self.expansion * inner_width)

    def forward(self, x):
        out = self.basic(x)
        out += self.shortcut(x)
        out = self.relu(self.bn0(out))
        return out

class ResNeXt_BinAct(nn.Module):
    def __init__(self, builder, layers, groups, bottleneck_width=64, expansion=2, num_classes=10):
        super(ResNeXt_BinAct, self).__init__()
        self.groups = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = 64
        self.expansion = expansion

        # self.conv0 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1)
        # self.bn0 = nn.BatchNorm2d(self.in_planes)
        # self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1=self._make_layer(num_blocks[0],1)
        # self.layer2=self._make_layer(num_blocks[1],2)
        # self.layer3=self._make_layer(num_blocks[2],2)
        # self.layer4=self._make_layer(num_blocks[3],2)
        # self.linear = nn.Linear(self.groups * self.bottleneck_width, num_classes)
        if args.first_layer_dense:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = builder.conv7x7(3, 64, stride=2, first_layer=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = builder.batchnorm(64)
        self.relu = (lambda: BiRealAct())()
        self.layer1 = self._make_layer(builder, 64, layers[0])
        self.layer2 = self._make_layer(builder, 64*(self.expansion+1), layers[1], stride=2)
        self.layer3 = self._make_layer(builder, 128*(self.expansion+1), layers[2], stride=2)
        self.layer4 = self._make_layer(builder, 256*(self.expansion+1), layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if args.last_layer_dense:
            self.fc = nn.Conv2d(512 * self.expansion, args.num_classes, 1)
        else:
            self.fc = builder.conv1x1(512 * self.expansion, num_classes)


    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        # out = self.pool0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.fc(out)
        out = out.view(out.size(0), -1)
        return out

    def _make_layer(self, builder, planes, num_blocks, stride=1):
        downsample = None
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_C(builder, planes, self.bottleneck_width, self.groups, stride, self.expansion))
            self.inplanes = self.expansion * self.bottleneck_width * self.groups
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)


# ResNet_BinAct {{{
class ResNet_BinAct(nn.Module):
    def __init__(self, builder, block, layers, num_classes=1000, base_width=64):
        self.inplanes = 64
        super(ResNet_BinAct, self).__init__()

        self.base_width = base_width
        if self.base_width // 64 > 1:
            print(f"==> Using {self.base_width // 64}x wide model")

        if args.first_layer_dense:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = builder.conv7x7(3, 64, stride=2, first_layer=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = builder.batchnorm(64)
        self.relu = (lambda: BiRealAct())()
        self.layer1 = self._make_layer(builder, block, 64, layers[0])
        self.layer2 = self._make_layer(builder, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(builder, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        if args.last_layer_dense:
            self.fc = nn.Conv2d(512 * block.expansion, args.num_classes, 1)
        else:
            self.fc = builder.conv1x1(512 * block.expansion, num_classes)

    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(
                self.inplanes, planes * block.expansion, stride=stride
            )
            dbn = builder.batchnorm(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x

# ResNet_BinAct }}}


# WideResNet_BinAct {{{
class WideResNet_BinAct(nn.Module):
    def __init__(self, builder, block, layers, num_classes=1000, base_width=64, widen_factor=1):
        self.inplanes = 64
        super(WideResNet_BinAct, self).__init__()

        self.base_width = base_width
        if self.base_width // 64 > 1:
            print(f"==> Using {self.base_width // 64}x wide model")

        if args.first_layer_dense:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = builder.conv7x7(3, 64, stride=2, first_layer=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = builder.batchnorm(64)
        self.relu = (lambda: BiRealAct())()
        self.layer1 = self._make_layer(builder, block, 64, layers[0])
        self.layer2 = self._make_layer(builder, block, 64*(widen_factor+1), layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, 128*(widen_factor+1), layers[2], stride=2)
        self.layer4 = self._make_layer(builder, block, 256*(widen_factor+1), layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        if args.last_layer_dense:
            self.fc = nn.Conv2d(256*(widen_factor+1) * block.expansion, args.num_classes, 1)
        else:
            self.fc = builder.conv1x1(256*(widen_factor+1) * block.expansion, num_classes)

    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(
                self.inplanes, planes * block.expansion, stride=stride
            )
            dbn = builder.batchnorm(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x

# WideResNet_BinAct }}}


# Imagenet Networks
def ResNet18_BinAct(pretrained=False):
    return ResNet_BinAct(get_builder(), BasicBlock, [2, 2, 2, 2], 1000)

def ResNet34_BinAct(pretrained=False):
    return ResNet_BinAct(get_builder(), BasicBlock, [3, 4, 6, 3], 1000)

def ResNet50_BinAct(pretrained=False):
    return ResNet_BinAct(get_builder(), Bottleneck, [3, 4, 6, 3], 1000)

def ResNet101_BinAct(pretrained=False):
    return ResNet_BinAct(get_builder(), Bottleneck, [3, 4, 23, 3], 1000)


def WideResNet18_2_BinAct(pretrained=False):
    return WideResNet_BinAct(get_builder(), BasicBlock, [2, 2, 2, 2], 1000, widen_factor=2)

def WideResNet18_3_BinAct(pretrained=False):
    return WideResNet_BinAct(get_builder(), BasicBlock, [2, 2, 2, 2], 1000, widen_factor=2.5)

def WideResNet34_2_BinAct(pretrained=False):
    return WideResNet_BinAct(get_builder(), BasicBlock, [3, 4, 6, 3], 1000, widen_factor=2)

def WideResNet34_3_BinAct(pretrained=False):
    return WideResNet_BinAct(get_builder(), BasicBlock, [3, 4, 6, 3], 1000, widen_factor=3)

def WideResNet50_2_BinAct(pretrained=False):
    return ResNet_BinAct(
        get_builder(), Bottleneck, [3, 4, 6, 3], num_classes=1000, base_width=64 * 2
    )



# CIFAR-10 Networks
def ResNext_BinAct(pretrained=False):
    return ResNeXt_BinAct(get_builder(), [1, 2, 6, 2], groups=4, expansion=2)


def cifarResNet18_BinAct(pretrained=False):
    return ResNet_BinAct(get_builder(), BasicBlock, [2, 2, 2, 2], 10)

def cifarWideResNet18_2_BinAct(pretrained=False):
    return ResNet_BinAct(get_builder(), BasicBlock, [2, 2, 2, 2], 10, widen_factor=2)

def cifarWideResNet18_3_BinAct(pretrained=False):
    return ResNet_BinAct(get_builder(), BasicBlock, [2, 2, 2, 2], 10, widen_factor=3)
