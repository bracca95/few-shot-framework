from torch import nn

from src.models.MLP.mlp_basic import MLP
from src.models.FSL.ProtoNet.protonet import ProtoNet
from src.utils.config_parser import Config


class Model(nn.Module):

    def __init__(self):
        super().__init__()

class ModelBuilder:

    @staticmethod
    def load_dataset(config: Config, out_classes) -> nn.Module:
        if config.fsl.model.lower() == "default":
            return ProtoNet()
        elif config.fsl.model.lower() == "mlp":
            return MLP(config.image_size * config.image_size)
        else:
            raise ValueError(f"fsl.model must be { {'default', 'mlp'} }. You wrote: {config.fsl.model}")