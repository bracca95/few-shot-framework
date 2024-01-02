import torch
from torch import nn


class ChannelAttention(nn.Module):

    def __init__(self, channel: int, reduction: int=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_pool = self.avg_pool(x).view(b, c)
        avg_out = self.se(avg_pool).view(b, c, 1, 1)
        
        max_pool = self.max_pool(x).view(b, c)
        max_out = self.se(max_pool).view(b, c, 1, 1)
        
        scale = self.sigmoid(avg_out + max_out)
        return x * scale.expand_as(x)
    

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size: int=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)       # channel dimension
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        
        cat_pool = torch.cat((avg_pool, max_pool), dim=1)
        attention = self.act(self.conv2d(cat_pool))

        return x * attention.expand_as(x), attention


class CBAMLayer(nn.Module):
    """CBAM Spatial Attention mechanism for CNNs

    SeeAlso:
        [papers with code](https://paperswithcode.com/method/spatial-attention-module)
        [youtube video](https://www.youtube.com/watch?v=1mjI_Jm4W1E)
        [linked ipynb](https://github.com/EscVM/EscVM_YT/blob/master/Notebooks/0%20-%20TF2.X%20Tutorials/tf_2_visual_attention.ipynb)
    """

    def __init__(self, in_channels: int, reduction_ratio: int=16, kernel_size: int=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

        # init weights
        self.init_weights(self)

    def forward(self, x):
        x_channel = self.channel_attention(x)
        x_spatial, att = self.spatial_attention(x_channel)
        return x_spatial, att
    
    @classmethod
    def init_weights(cls, model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            else:
                pass