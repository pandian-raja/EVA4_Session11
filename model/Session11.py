
import torch
import torch.nn as nn
import torch.nn.functional as F

class Layer(nn.Module):

    def __init__(self, in_planes, planes, resBlock=False):
        super(Layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_planes, out_channels = planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if resBlock:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels = planes, out_channels = planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels = planes, out_channels = planes, kernel_size=3, stride=1,padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.pool(self.conv1(x))),inplace=False)
        r1 = self.shortcut(out)
        out = out + r1
        return out


class Session11(nn.Module):
    def __init__(self, num_classes=10):
        super(Session11, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels =64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(Layer(64, 128,True))
        self.layer2 = nn.Sequential(Layer(128, 256,False))
        self.layer3 = nn.Sequential(Layer(256, 512,True))
        self.linear = nn.Linear(512, num_classes,bias=False)

    # def _make_layer(self, block, planes, num_blocks, resBlock):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)),inplace=False)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.max_pool2d(input = out, kernel_size=4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # x = out.view(-1, 10)
        return F.log_softmax(out, dim=-1)
        

# net = Session11()
# print(net)
# summary(net.cuda(), input_size=(3, 32, 32))
