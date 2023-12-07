from torch import nn
from src.models.model import Model
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
