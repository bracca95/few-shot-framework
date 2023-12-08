import torch

from torch import nn
from typing import Optional

from src.models.model import Model
from src.utils.config_parser import Config
from lib.glass_defect_dataset.src.utils.tools import Logger
from lib.glass_defect_dataset.config.consts import General as _CG


class ClassificationHead(Model):

    def __init__(self, config: Config, extractor: Model, out_class: int):
        super().__init__(config)

        self.extractor = extractor.to(_CG.DEVICE)
        self.backbone_features: int = self.extractor.get_out_size(1)
        if config.model.freeze:
            Logger.instance().debug("freezing extractor layers")
            list(map(lambda param: setattr(param, 'requires_grad', False), self.extractor.parameters()))

        self.linear = nn.Sequential(
            nn.Linear(self.backbone_features, out_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.extractor(x)
        return self.linear(out)
    

class ManifoldMixup(ClassificationHead):

    def __init__(self, config: Config, extractor: Model, out_class: int=256):
        super().__init__(config, extractor, out_class)

        self.linear = nn.Sequential(
            nn.Linear(self.backbone_features, out_class),
            nn.BatchNorm1d(out_class),
            nn.ReLU(),
            nn.Linear(out_class, out_class)
        )

    def forward(self, *args):
        assert len(args) == 1 or len(args) == 3, f"Model can accept 1 or 3 inputs. You passed {len(args)}."
        
        x = args[0]
        shuffle: Optional[torch.Tensor] = None
        lam: Optional[float] = None
        
        if len(args) > 1:
            assert type(args[1]) is torch.Tensor, "pass values as input (Tensor), shuffle (Tensor), lambda (float)"
            assert type(args[2]) is float, "pass values as input (Tensor), shuffle (Tensor), lambda (float)"
            _, shuffle, lam = args
        
        out = self.extractor(x)
        out = self.mixup(out, shuffle, lam)
        out = self.linear(out)

        return out

    @staticmethod
    def mixup(x: torch.Tensor, shuffle: Optional[torch.Tensor], lam: Optional[float]) -> torch.Tensor:
        # https://www.kaggle.com/code/hocop1/manifold-mixup-using-pytorch
        if shuffle is not None and lam is not None:
            x = lam * x + (1 - lam) * x[shuffle]
        return x
    

class SimCLRProj(ClassificationHead):

    def __init__(self, config: Config, extractor: Model, out_class: int):
        super().__init__(config, extractor, out_class)
        self.linear = nn.Sequential(
            nn.Linear(self.backbone_features, out_class),
            nn.ReLU(),
            nn.Linear(out_class, out_class)
        )

    def forward(self, x):
        return self.linear(x)