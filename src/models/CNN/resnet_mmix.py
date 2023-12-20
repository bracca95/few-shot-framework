# https://github.com/DaikiTanak/manifold_mixup/blob/master/resnet_mixup.py
import torch

from abc import ABC
from torch import nn
from typing import Callable, Optional, List

from src.models.model import Model
from src.models.attention.squeeze_excitation import SELayer
from src.utils.config_parser import Config


def conv3x3(in_planes: int, out_planes: int, stride: int=1, bias: bool=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

def conv1x1(in_planes: int, out_planes: int, stride: int=1, bias: bool=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class ResidualOrBottleneck(ABC, nn.Module):

    EXPANSION: int = 1
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ...

class SEBasicBlock(ResidualOrBottleneck):
    EXPANSION = 1

    def __init__(self, inplanes: int, planes: int, stride: int=1, downsample: Optional[Callable]=None, reduction: int=16):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.reduction = reduction

        # bn - 3*3 conv - bn - relu - dropout - 3*3 conv - bn - add
        # https://arxiv.org/pdf/1610.02915.pdf
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.drop = nn.Dropout2d(p=0.3)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        # bn - 3*3 conv - bn - relu - dropout - 3*3 conv - bn - add
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out
    

class SEBottleneck(ResidualOrBottleneck):
    EXPANSION = 4

    def __init__(self, inplanes: int, planes: int, stride: int=1, downsample: Optional[Callable]=None, reduction: int=16):
        super(SEBottleneck, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * self.EXPANSION, reduction)
        self.downsample = downsample
        self.stride = stride

        # bn - 1*1conv - bn - relu - 3*3conv - bn - relu - 1*1conv - bn
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv1x1(inplanes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn3 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.EXPANSION, stride=1)
        self.bn4 = nn.BatchNorm2d(planes * self.EXPANSION)

    def forward(self, x):
        residual = x
        # bn - 1*1conv - bn - relu - 3*3conv - bn - relu - 1*1conv - bn
        # This architecture is proposed in Deep Pyramidal Residual Networks.

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn4(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class ResNetMixup(Model):
    """ResNet with Manifold-Mixup
    
    SeeAlso:
        https://arxiv.org/pdf/1806.05236.pdf
    """

    WIDEN_FACTOR: int = 1

    def __init__(
            self,
            config: Config,
            block: ResidualOrBottleneck,
            layers: List[int],
            num_classes: int=256,
            zero_init_residual=True
    ):
        super().__init__(config)
        self.inplanes = 64
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64*self.WIDEN_FACTOR, layers[0])
        self.layer2 = self._make_layer(block, 128*self.WIDEN_FACTOR, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256*self.WIDEN_FACTOR, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512*self.WIDEN_FACTOR, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.EXPANSION * self.WIDEN_FACTOR, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, SEBottleneck):
                    nn.init.constant_(m.bn4.weight, 0)
                elif isinstance(m, SEBasicBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, layers, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.EXPANSION:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.EXPANSION, stride),
                nn.BatchNorm2d(planes * block.EXPANSION),
            )

        resnet_layers = []
        resnet_layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.EXPANSION
        for _ in range(1, layers):
            resnet_layers.append(block(self.inplanes, planes))

        return nn.Sequential(*resnet_layers)

    def forward(self, *args):
        assert len(args) == 1 or len(args) == 3, f"Model can accept 1 or 3 inputs. You passed {len(args)}."
        
        x = args[0]
        shuffle: Optional[torch.Tensor] = None
        lam: Optional[float] = None
        layer_mix: Optional[int] = None
        
        if len(args) > 1:
            assert type(args[1]) is torch.Tensor, "pass values as input (Tensor), shuffle (Tensor), lambda (float)"
            assert type(args[2]) is float, "pass values as input (Tensor), shuffle (Tensor), lambda (float)"
            _, shuffle, lam = args
            layer_mix = torch.randint(0, 6, (1,), dtype=torch.int).item()
        
        out = x

        if layer_mix == 0:
            out = self.mixup(out, shuffle, lam)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

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

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        if layer_mix == 5:
            out = self.mixup(out, shuffle, lam)

        return out
        
    @staticmethod
    def mixup(x: torch.Tensor, shuffle: Optional[torch.Tensor], lam: Optional[float]) -> torch.Tensor:
        # https://www.kaggle.com/code/hocop1/manifold-mixup-using-pytorch
        if shuffle is not None and lam is not None:
            x = lam * x + (1 - lam) * x[shuffle]
        return x