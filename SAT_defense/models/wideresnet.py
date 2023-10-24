import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['wideResNet', 'wideResnet28_10']


def conv3x3(c_in, c_out, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(c_in, c_out, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1, dropout_rate=0.3, downsample=None):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(c_in, c_out, stride)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.downsample = downsample

        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = conv3x3(c_out, c_out)
        self.stride = stride

    def forward(self, x):

        identity = x
        out = self.bn1(x)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return out


class wideResNet(nn.Module):

    def __init__(self, block, layers, widen, dropout_rate=0.3, num_classes=10):
        super(wideResNet, self).__init__()

        self.conv1 = conv3x3(3, 16)

        self.layer1 = self._make_layer(
            block, 16, 16*widen, layers[0], dropout_rate)
        self.layer2 = self._make_layer(
            block, 16*widen, 32*widen, layers[1], dropout_rate, stride=2)
        self.layer3 = self._make_layer(
            block, 32*widen, 64*widen, layers[2], dropout_rate, stride=2)

        self.batch_norm = nn.BatchNorm2d(64*widen)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*widen, num_classes)

        self.output_num = 4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.1)

    def _make_layer(self, block, c_in, c_out, blocks, dropout_rate, stride=1):
        downsample = None
        if c_in != c_out or stride != 1:
            downsample = nn.Sequential(
                conv1x1(c_in*block.expansion, c_out * block.expansion, stride))
            # downsample = nn.Sequential(nn.MaxPool2d(stride,stride),nn.ConstantPad3d([0,0,0,0,0,(c_out - c_in)* block.expansion],0))#functional.pad(x,[0,0,0,0,0,(c_out - c_in)* block.expansion]))
        layers = []
        layers.append(block(c_in*block.expansion, c_out,
                      stride, dropout_rate, downsample))
        for _ in range(1, blocks):
            layers.append(block(c_out * block.expansion,
                          c_out, dropout_rate=dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)

        x1 = self.layer1(x)

        x2 = self.layer2(x1)

        x3 = self.layer3(x2)

        x = self.batch_norm(x3)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def wideResnet28_10(num_classes: int = 10):
    """Constructs a wideResnet28_10 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = wideResNet(BasicBlock, [4, 4, 4], num_classes)
    return model


if __name__ == '__main__':
    import torch
    x = wideResnet28_10()
    y = torch.rand(1, 3, 32, 32)
    for mid in x(y):
        print(mid.shape)

import random
print(random.randint(0,100))