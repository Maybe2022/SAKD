from spikingjelly.clock_driven import neuron, surrogate, cu_kernel_opt, layer, functional
import torch.nn as nn
import torch
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)




def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, T=4):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample


        self.spike1 = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), backend='cupy',decay_input=False)
        # self.spike2 = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), backend='cupy',decay_input=False)
        # print(surrogate.atan.gama)
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
        # out = self.spike2(out)
        return out


def resotre(x):
    T, B, C, H, W = x.shape
    x = x.reshape(-1, C, H, W)
    return x


def divide(x, T):
    B, C, H, W = x.shape
    x = x.reshape(T, B // T, C, H, W)
    return x


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, T=4):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.spike1 = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), backend='cupy', decay_input=False,
                                              )
        self.spike2 = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), backend='cupy', decay_input=False,
                                              )
        # self.spike3 = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), backend='cupy', decay_input=False,
        #
        #                                       )


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
        out = self.spike2(out)

        out = resotre(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = divide(out, T)

        if self.downsample is not None:
            identity = resotre(identity)
            identity = self.downsample(identity)
            identity = divide(identity, T)

        out += identity
        # out = self.relu(out)
        return out
        # return self.spike3(out)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.T = T

        self.inplanes = 64
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

        self.layer1 = self._make_layer(block, 64, layers[0], T=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], T=4)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], T=6)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], T=8)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #



        # self.connection1 = nn.Conv2d(64 , 128, 3, 2, 1, bias=False)
        # self.connection2 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        # self.connection3 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)

        self.spike = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(),backend='cupy', decay_input=False,detach_reset=True)

        self.layer = layers



        self.act = []
        for layer in layers:
            for i in range(layer):
                self.act.append(neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(), backend='cupy', decay_input=False,
                                             detach_reset=True))
        self.act = nn.ModuleList(self.act)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        self.transform1 = transform(64 * block.expansion)
        self.transform2 = transform(128 * block.expansion)
        self.transform3 = transform(256 * block.expansion)
        self.transform4 = transform(512 * block.expansion)




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
    def forward(self, input ):

        # if not  self.training:
        x = input.unsqueeze(0)
        x = x.repeat(self.T , 1, 1, 1, 1)

        feature = []
        x = resotre(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = divide(x, self.T)
        x = self.spike(x)
        h = 1



        for blk in self.layer1:
            x = blk(x)
            x = self.act[h - 1](x)
            h += 1

        if  self.training :
            feature.append(self.transform1(x.mean(0)))


        for blk in self.layer2:
            x = blk(x)
            x = self.act[h - 1](x)
            h += 1

        if self.training:
            feature.append(self.transform2(x.mean(0)))

        for blk in self.layer3:
            x = blk(x)
            x = self.act[h - 1](x)
            h += 1
        if self.training:
            feature.append(self.transform3(x.mean(0)))

        for blk in self.layer4:
            x = blk(x)
            x = self.act[h - 1](x)
            h += 1
        if self.training:
            feature.append(self.transform4(x.mean(0)))


        ####norm
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)

        if self.training:
            return x.mean(0),feature

        return x.mean(0)






def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18__(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34__(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet50__(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet101__(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152__(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


class transform(nn.Module):
    def __init__(self, channel):
        super(transform, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel),
        )

    def forward(self, x):

        return self.net(x)

