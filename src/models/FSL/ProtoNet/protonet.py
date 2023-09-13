import torch.nn as nn

from src.models.model import Model
from src.utils.config_parser import Config

def conv_block(in_channels: int, out_channels: int):
    #returns a block conv-bn-relu-pool
    
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNet(Model):
    """Model as described in the reference paper,
    
    SeeAlso:
        https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    """
    
    def __init__(self, config: Config, hid_dim: int=64, z_dim: int=64):
        super().__init__(config)
        in_chans = 1 if config.dataset.dataset_mean is None else len(config.dataset.dataset_mean)
        self.encoder = nn.Sequential(
            conv_block(in_chans, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1) # with img_size 105, output size is: (batch_size, 64*6*6)