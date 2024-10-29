import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.special import softmax

from args import args as parser_args

affine_par = True


def sparseFunction(x, s, activation=torch.relu, f=torch.sigmoid):
    return torch.sign(x)*activation(torch.abs(x)-f(s))

def initialize_sInit():

    if parser_args.sInit_type == "constant":
        # return parser_args.sInit_value*torch.ones([1, 1])
        return parser_args.sInit_value*torch.tensor(1)


class STRConv(nn.Conv2d):
    def __init__(self, *args, sparseThreshold, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = torch.relu

        if parser_args.sparse_function == 'sigmoid':
            self.f = torch.sigmoid
            self.sparseThreshold = sparseThreshold
        #     self.sparseThreshold = nn.Parameter(initialize_sInit())
        # else:
        #     self.sparseThreshold = nn.Parameter(initialize_sInit())
    
    def forward(self, x):
        # In case STR is not training for the hyperparameters given in the paper, change sparseWeight to self.sparseWeight if it is a problem of backprop.
        # However, that should not be the case according to graph computation.
        # self.weight.data.mul_(torch.relu(torch.abs(self.weight.data) - self.f(self.sparseThreshold)))
        # x = F.conv2d(
        #     self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        # )
        sparseWeight = sparseFunction(self.weight, self.sparseThreshold, self.activation, self.f)
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        # different between F.conv2d and nn.conv2d,
        # ref: https://discuss.pytorch.org/t/difference-results-with-torch-nn-conv2d-and-torch-nn-functional-conv2d/69231
        return x

    def getSparsity(self, f=torch.sigmoid):
        sparseWeight = sparseFunction(self.weight, self.sparseThreshold,  self.activation, self.f)
        temp = sparseWeight.detach().cpu()
        temp[temp!=0] = 1
        return (100 - temp.mean().item()*100), temp.numel(), f(self.sparseThreshold).item()





# ################################################
# ############## ResNet Modules ##################
# ################################################

def conv3x3(in_channels, out_channels, stride=1, dilation=1, sparseThreshold=None):
    "3x3 convolution with padding"

    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return STRConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=full_padding, dilation=dilation, bias=False, sparseThreshold=sparseThreshold)


# No projection: identity shortcut
# conv -> bn -> relu -> conv -> bn
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, sparseThreshold=None):
        super(BasicBlock, self).__init__()
        self.planes = planes

        # 4 level channel usage: 0 -- 0%; 1 -- 25 %; 2 -- 50 %; 3 -- 100%
        self.keep_channels = (planes * np.cumsum([0, 0.25, 0.25, 0.5])).astype('int')
        self.keep_masks = []
        for kc in self.keep_channels:
            mask = np.zeros([1, planes, 1, 1])
            mask[:, :kc] = 1
            self.keep_masks.append(mask)
        self.keep_masks = torch.from_numpy(np.concatenate(self.keep_masks)).float()

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation, sparseThreshold=sparseThreshold)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation, sparseThreshold=sparseThreshold)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)

    def forward(self, x, keep=None):
        # keep: [batch_size], int
        cuda_device = x.get_device()
        out = self.conv1(x)
        out = self.bn1(out)

        # used for deep elastic
        if keep is not None:
            keep = keep.long()
            bs, h, w = out.shape[0], out.shape[2], out.shape[3]
            # mask: [batch_size, c, 1, 1]
            mask = self.keep_masks[keep].to(cuda_device)
            # mask: [batch_size, c, h, w]
            mask = mask.repeat(1, 1, h, w)
            out = out * mask

        out = self.relu(out)
        out = self.conv2(out)
        y = self.bn2(out)
        return y

# No projection: identity shortcut and atrous
'''
class Bottleneck(nn.Module):
    expansion = 4

    # |----------------------------------------------------------------|
    # 1x1 conv -> bn -> relu -> 3x3 conv -> bn -> relu -> 1x1 conv -> bn -> relu
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = STRConv(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = STRConv(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = STRConv(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out
'''


class Deeplab_ResNet_Backbone(nn.Module):
    # the init arguments are for resnet34
    def __init__(self, block=BasicBlock, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super(Deeplab_ResNet_Backbone, self).__init__()
        self.sparseThreshold = nn.Parameter(initialize_sInit())
        self.conv1 = STRConv(3, 64, sparseThreshold=self.sparseThreshold, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change

        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []

        for idx, (filt_size, num_blocks, stride, dilation) in enumerate(zip(filt_sizes, layers, strides, dilations)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride, dilation=dilation)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        #
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, STRConv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                STRConv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, sparseThreshold=self.sparseThreshold),
                nn.BatchNorm2d(planes * block.expansion, affine = affine_par))
            # for i in downsample._modules['1'].parameters():
            #     i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, sparseThreshold=self.sparseThreshold))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, sparseThreshold=self.sparseThreshold))

        return layers, downsample

    def forward(self, x, policy=None):
        if policy is None:
            # forward through the all blocks without dropping
            x = self.seed(x)
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    # apply the residual skip out of _make_layers_

                    residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    x = F.relu(residual + self.blocks[segment][b](x))

        else:
            # do the block dropping
            x = self.seed(x)
            t = 0
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    residual = self.ds[segment](x) if b == 0 and self.ds[segment] is not None else x
                    fx = F.relu(residual + self.blocks[segment][b](x))
                    if policy.ndimension() == 2:
                        x = fx * policy[t, 0] + residual * policy[t, 1]
                    elif policy.ndimension() == 3:
                        x = fx * policy[:, t, 0].contiguous().view(-1, 1, 1, 1) + residual * policy[:, t, 1].contiguous().view(-1, 1, 1, 1)
                    elif policy.ndimension() == 1:
                        x = fx * policy[t] + residual * (1-policy[t])
                    t += 1
        return x
        # return [x, x, x]        # in this forward case, we need to duplicate the layers for n-task
