from torchvision.models.vgg import vgg16
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from collections import OrderedDict

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layers = self._make_layers()

    def forward(self, x):
        y = self.layers(x)
        return y

    def _make_layers(self):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.ReLU(True)]
                in_channels = x

            return nn.Sequential(*layers)



class L2Norm(nn.Module):
    # scale/weight是一个需要训练的参数
    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)
        # print(self.weight)
        # print(self.weight.shape)
        # print(self.weight[None, :, None, None].shape) # torch.Size([1, 512, 1, 1])

    def reset_parameters(self, scale):
        nn.init.constant(self.weight, scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None, :, None, None]
        #print(scale)
        return scale * x


# x = L2Norm(512, 20)
