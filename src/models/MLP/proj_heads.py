import torch

from torch import nn, Tensor
from typing import Tuple, Optional
from torchvision.transforms.functional import rotate

from src.models.model import Model
from src.models.pretrained.classification_head import ManifoldMixup
from src.utils.config_parser import Config

class ProjHead(Model):

    def __init__(self, config: Config, in_feat: int, out_feat: int=256):
        super().__init__(config)

        self.proj = nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.BatchNorm1d(out_feat),
            nn.ReLU(),
            nn.Linear(out_feat, out_feat)
        )

    def forward(self, x):
        out = self.proj(x)
        return out


class RotationLayer(Model):

    def __init__(self, config: Config, in_feat: int, out_classes: int=4):
        super().__init__(config)

        self.proj_1 = nn.Sequential(
            nn.Linear(in_feat, in_feat // 2),
            nn.BatchNorm1d(in_feat // 2),
            nn.ReLU(inplace=True)
        )

        self.proj_2 = nn.Sequential(
            nn.Linear(in_feat // 2, in_feat // 4),
            nn.BatchNorm1d(in_feat // 4),
            nn.ReLU(inplace=True)
        )

        self.classify = nn.Linear(in_feat // 4, out_classes)

    def forward(self, x):
        out = self.proj_1(x)
        out = self.proj_2(out)
        out = self.classify(out)
        return out
    
    @staticmethod
    def rotate_batch(x: Tensor, one_hot: bool, shuffle: Optional[Tensor]=None, lam: Optional[float]=None) -> Tuple[Tensor, Tensor]:
        batch_size = x.size(0)

        # get indices of images tha will undergo specific rotations
        indices = torch.randperm(batch_size)
        rot0_idx, rot90_idx, rot180_idx, rot270_idx = torch.chunk(indices, 4, dim=0)

        # rotate corresponding indices
        rot0 = x[rot0_idx]
        rot90 = rotate(x[rot90_idx], 90.0)
        rot180 = rotate(x[rot180_idx], 180.0)
        rot270 = rotate(x[rot270_idx], 270.0)

        # keep the original order
        x[rot0_idx] = rot0
        x[rot90_idx] = rot90
        x[rot180_idx] = rot180
        x[rot270_idx] = rot270

        y = torch.zeros(batch_size, device=x.device, dtype=torch.int64, requires_grad=False)
        y[rot0_idx] = int(0)
        y[rot90_idx] = int(1)
        y[rot180_idx] = int(2)
        y[rot270_idx] = int(3)

        if one_hot:
            y = y.type(torch.int64)     # indices must be integers
            y = Model.one_hot_encoding(y, n_classes=4, dtype=torch.float)
            if shuffle is not None and lam is not None:
                y = ManifoldMixup.mixup(y, shuffle, lam)

        return x, y
    
    @staticmethod
    def rotation_loss(x: Tensor, y: Tensor):
        log_p_y = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -(y * log_p_y + 1e-10).sum(dim=-1).mean()

        return loss