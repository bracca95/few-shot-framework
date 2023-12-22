import torch
import torch.nn as nn

from typing import Callable, Optional, List

from src.models.model import Model
from src.models.attention.squeeze_excitation import SELayer
from src.models.attention.cbam import CBAMLayer
from src.utils.config_parser import Config


def conv3x3(in_planes: int, out_planes: int):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


def norm_layer(planes: int):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes: int, planes: int, downsample: Callable, attention: bool):
        super().__init__()
        self.attention = attention

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)
        self.att_channel = SELayer(planes)
        self.att_spatial = CBAMLayer()

        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        # conv layer 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # conv layer 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # conv layer 3 + optional attention
        out = self.conv3(out)
        out = self.bn3(out)
        if self.attention:
            out = self.att_channel(out)
            out = self.att_spatial(out)

        identity = self.downsample(x)
        out += identity
        
        # for every residual block
        out = self.relu(out)
        out = self.maxpool(out)

        return out


class ResNet12(Model):
    """ResNet12 architecture

    ResNet12 can be found in many different formats. This is found in [1] and [2], with the addition of attention
    layers to adapt to our needs. This architecture is ideal for 84x84 image size (miniImagenet)
    
    As stated in [2]:
    ResNet12 contains 4 residual blocks and each block has 3 CONV layers with 3x3 kernels. The first two CONV layers are
    followed by a batch normalization and a ReLU nonlinearity, and the last CONV layer is followed by a batch 
    normalization and a skip connection which contains a 1x1 convolution. A ReLU nonlinearity and a 2x2 max-pooling 
    layer are applied at the end of each residual block after a skip connection. The number of convolution filters for 
    4 residual blocks is set to be 64, 128, 256, 512 in the increasing order of depth.

    SeeAlso:
        [1][Chen et al.](https://github.com/yinboc/few-shot-meta-baseline/blob/master/models/resnet12.py)
        [2][Jia et al.](https://www.sciencedirect.com/science/article/pii/S095219762301480X)
    """

    def __init__(self, config: Config, attention: bool, channels: List[int]=[64, 128, 256, 512]):
        super().__init__(config)
        self.attention = attention

        self.inplanes = 3

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample, self.attention)
        self.inplanes = planes
        
        return block

    def forward(self, *args):
        # structured for manifold mixup
        assert len(args) == 1 or len(args) == 3, f"Model can accept 1 or 3 inputs. You passed {len(args)}."
        
        x = args[0]
        shuffle: Optional[torch.Tensor] = None
        lam: Optional[float] = None
        layer_mix: Optional[int] = None
        
        if len(args) > 1:
            assert type(args[1]) is torch.Tensor, "pass values as input (Tensor), shuffle (Tensor), lambda (float)"
            assert type(args[2]) is float, "pass values as input (Tensor), shuffle (Tensor), lambda (float)"
            _, shuffle, lam = args
            layer_mix = torch.randint(0, 5, (1,), dtype=torch.int).item()
        
        out = x
        if layer_mix == 0: # mixup at image level
            out = self.mixup(out, shuffle, lam)

        out = self.layer1(out)
        if layer_mix == 1:
            out = self.mixup(out, shuffle, lam)

        out = self.layer2(out)
        if layer_mix == 2:
            out = self.mixup(out, shuffle, lam)

        out = self.layer3(out)
        if layer_mix == 3:
            out = self.mixup(out, shuffle, lam)

        out = self.layer4(out)
        if layer_mix == 4:
            out = self.mixup(out, shuffle, lam)

        descriptors = out
        out = self.avg_pool(out).view(out.size(0), -1)
        
        return out, descriptors
    
    @staticmethod
    def mixup(x: torch.Tensor, shuffle: Optional[torch.Tensor], lam: Optional[float]) -> torch.Tensor:
        # https://www.kaggle.com/code/hocop1/manifold-mixup-using-pytorch
        if shuffle is not None and lam is not None:
            x = lam * x + (1 - lam) * x[shuffle]
        return x