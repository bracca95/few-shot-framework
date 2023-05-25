from torch import nn

from src.models.MLP.mlp_basic import MLP
from src.models.CNN.cnn_basic import CNN
from src.models.FSL.ProtoNet.protonet import ProtoNet
from src.models.pretrained.extractors import TimmFeatureExtractor
from src.utils.config_parser import Config


class Model(nn.Module):

    def __init__(self):
        super().__init__()

class ModelBuilder:

    @staticmethod
    def load_model(config: Config, out_classes) -> nn.Module:
        if config.fsl.model.lower() == "default":
            return ProtoNet()
        elif config.fsl.model.lower() == "mlp":
            return MLP(config.image_size * config.image_size)
        elif config.fsl.model.lower() == "cnn":
            return CNN()
        elif config.fsl.model.lower() in ("resnet50"):
            return TimmFeatureExtractor(config.fsl.model.lower(), in_chans=1, pooled=True, mean=config.dataset_mean, std=config.dataset_std)
        else:
            raise ValueError(f"fsl.model must be { {'default', 'mlp', 'cnn', 'resnet50'} }. You wrote: {config.fsl.model}")