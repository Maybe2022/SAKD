import torch
import torch.nn as nn
# from .layer import *
from spikingjelly.clock_driven import neuron, surrogate, cu_kernel_opt, layer, functional



# from utils import mask


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class conv3d(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(conv3d, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv3d(in_channel, out_channel, 3, 1, 1, bias=False),
#             nn.BatchNorm3d(out_channel),
#         )
#         self.spike = LIFSpike()
#
#     def forward(self, x):
#         # T,B,C,H,W = x.shape
#         # x = x.reshape(T,B,4,C//4,H,W).permute(1,2,3,0,4,5).reshape(-1,C//4,T,H,W)
#         # x = self.net(x).reshape(B,4,C//4,T,H,W).permute(3,0,1,2,4,5).reshape(T,B,C,H,W)
#         x = self.net(x.permute(1, 2, 0, 3, 4)).permute(2, 0, 1, 3, 4)
#         return self.spike(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=16, dilation=1, norm_layer=None, T=4):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 16:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        # self.spike = LIFSpike()

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.spike1 = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), backend='cupy', decay_input=False)
        self.spike2 = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), backend='cupy', decay_input=False)

        self.T = T

    def forward(self, x):
        T = x.shape[0]
        identity = x

        out = resotre(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = divide(out, T)

        out = self.spike1(out)
        out = resotre(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = divide(out, T)

        if self.downsample is not None:
            identity = resotre(identity)
            identity = self.downsample(identity)
            identity = divide(identity, T)

        out = out + identity

        # return out
        return self.spike2(out)


def resotre(x):
    T, B, C, H, W = x.shape
    x = x.reshape(-1, C, H, W)
    return x


def divide(x, T):
    B, C, H, W = x.shape
    x = x.reshape(T, B // T, C, H, W)
    return x


def spatial_divide(x, patch_size=8):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size).permute(2, 4, 0, 1, 3, 5).reshape(-1,
                                                                                                                    B,
                                                                                                                    C,
                                                                                                                    patch_size,
                                                                                                                    patch_size)
    # x = x.reshape(B, C, H // 2, 2, W // 2, 2).permute(2, 4, 0, 1, 3, 5).reshape(-1, B, C, H // 2, W // 2)
    return x


def channel_divide(x, T):
    B, C, H, W = x.shape
    x = x.reshape(B, T, C // T, H, W).permute(1, 0, 2, 3, 4)
    # x = x.reshape(B,C//16,16,H,W).permute(1,0,2,3,4)
    return x


def spatial_restore(x):
    T, B, C, H, W = x.shape
    rotio = int(pow(T, 0.5))
    x = x.reshape(rotio, rotio, B, C, H, W).permute(2, 3, 0, 4, 1, 5).reshape(B, C, H * rotio, W * rotio)
    return x


def channel_restore(x):
    T, B, C, H, W = x.shape
    x = x.permute(1, 0, 2, 3, 4).reshape(B, -1, H, W)

    return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=16, replace_stride_with_dilation=None,
                 norm_layer=None, T=4,step=3,channel=[64,128,256,512]):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.step = step
        self.T = T

        self.inplanes = channel[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, channel[1], layers[0], T=2)
        self.layer2 = self._make_layer(block, channel[2], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], T=4)
        self.layer3 = self._make_layer(block, channel[3], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], T=6)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel[3] * block.expansion, 256)
        self.fc_ = nn.Linear(256, num_classes)

        self.transform1 = transform(channel[1])
        self.transform2 = transform(channel[2])
        self.transform3 = transform(channel[3])

        self.spike = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), backend='cupy', decay_input=False)

        self.spike_ = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), backend='cupy', decay_input=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, T=4):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, T=T))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, T=T + i))

        # return nn.Sequential(*layers)
        return nn.ModuleList(layers)

    # def _forward_impl(self, x):
    def forward(self, x):

        # if not self.training:
        x.unsqueeze_(0)
        x = x.repeat(self.T, 1, 1, 1, 1)

        feature = []

        x = resotre(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = divide(x, self.T)
        x = self.spike(x)


        for blk in self.layer1:
            x = blk(x)
        if self.training :
                feature.append(self.transform1(torch.mean(x ,dim=0)))

        for blk in self.layer2:
            x = blk(x)

        if self.training :
                feature.append(self.transform2(torch.mean(x ,dim=0)))
        for blk in self.layer3:
            x = blk(x)
        if self.training :
                feature.append(self.transform3(torch.mean(x ,dim=0)))

        x = self.avgpool(x)

        x = torch.flatten(x, 2)

        out = self.fc(x)
        out = self.spike_(out)
        out = self.fc_(out)
        out = torch.mean(out,dim=0)

        if self.training:
            return out, feature
        return out



def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet19_(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [3, 3, 2], pretrained, progress,
                   **kwargs)

class transform(nn.Module):
    def __init__(self, channel):
        super(transform, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channel),
            # nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
