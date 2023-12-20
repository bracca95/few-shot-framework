import torch
from torch import nn


class CBAMLayer(nn.Module):
    """CBAM Spatial Attention mechanism for CNNs

    SeeAlso:
        [papers with code](https://paperswithcode.com/method/spatial-attention-module)
        [youtube video](https://www.youtube.com/watch?v=1mjI_Jm4W1E)
        [linked ipynb](https://github.com/EscVM/EscVM_YT/blob/master/Notebooks/0%20-%20TF2.X%20Tutorials/tf_2_visual_attention.ipynb)
    """
    def __init__(self, kernel_size: int=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=3, bias=False)
        self.act = nn.Sigmoid()

        # init weights
        nn.init.kaiming_normal_(self.conv2d.weight, mode='fan_out', nonlinearity='sigmoid')

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)       # channel dimension
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        
        cat_pool = torch.cat((avg_pool, max_pool), dim=1)
        attention = self.act(self.conv2d(cat_pool))

        return x * attention.expand_as(x)