import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, step = 3,num_classes=10, dim=[3, 3, 3],channel = [16,16,32,64]):
        super(ResNet, self).__init__()
        self.step = step
        self.inchannel = channel[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channel[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, channel[1], dim[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, channel[2], dim[1], stride=2)
        self.layer3 = self.make_layer(ResidualBlock, channel[3], dim[2], stride=2)
        self.fc = nn.Linear(channel[3], num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        feature = []
        # feature.append(out)
        for blk in self.layer1:
            out = blk(out)
        feature.append(out)

        for blk in self.layer2:
            out = blk(out)
        feature.append(out)

        for blk in self.layer3:
            out = blk(out)
        feature.append(out)

        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, feature
        # return feature


def resnet19(num_classes=10,step = 3):
    return ResNet(ResidualBlock, step=step,num_classes=num_classes, dim=[3, 3, 2],channel=[64,128,256,512])
