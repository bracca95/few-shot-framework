from torch import nn

from src.models.model import Model
from src.utils.config_parser import Config
from lib.glass_defect_dataset.src.utils.tools import Logger


class ClassificationHead(Model):

    def __init__(self, config: Config, extractor: Model, out_class: int):
        super().__init__(config)

        self.extractor = extractor
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