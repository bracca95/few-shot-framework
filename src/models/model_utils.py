from src.models.model import Model
from src.models.MLP.mlp_basic import MLP
from src.models.CNN.cnn_basic import CNN
from src.models.CNN.cnn_105 import CNN105
from src.models.FSL.ProtoNet.protonet import ProtoNet
from src.models.pretrained.extractors import TimmFeatureExtractor
from src.models.pretrained.classification_head import Head
from src.utils.config_parser import Config
    

class ModelBuilder:

    @staticmethod
    def load_model(config: Config, out_classes) -> Model:
        model_name = config.model.model_name.lower().replace(" ", "")
        
        # standard classification models
        if "compare" in model_name:
            model_name_stripped = model_name.replace("compare", "")
            if model_name_stripped == "cnn":
                extractor = extractor = CNN(config)
            elif model_name_stripped == "cnn105":
                extractor = CNN105(config)
            elif model_name_stripped in ("resnet50", "hrnet_w18", "vit_tiny_patch16_224"):
                extractor = TimmFeatureExtractor(
                    config,
                    model_name_stripped,
                    in_chans=1,
                    pooled=True,
                    mean=config.dataset.dataset_mean,
                    std=config.dataset.dataset_std
                )
            else:
                raise ValueError("No 'compare' model available.")
            return Head(config, extractor, out_classes)
            
        # few-shot learning models
        if model_name == "default":
            return ProtoNet(config)
        if model_name == "mlp":
            return MLP(config, config.dataset.image_size * config.dataset.image_size)
        if model_name == "cnn":
            return CNN(config)
        if model_name == "cnn105":
            return CNN105(config)
        if model_name in ("resnet50", "hrnet_w18", "vit_tiny_patch16_224"):
            return TimmFeatureExtractor(
                config,
                model_name,
                in_chans=1,
                pooled=True,
                mean=config.dataset.dataset_mean,
                std=config.dataset.dataset_std
            )
        
        # if the inserted string is wrong
        raise ValueError(
            f"fsl.model must be" +
            f"{ {'proto_default', 'mlp', 'cnn', 'cnn105', 'resnet50', 'hrnet_w18', 'vit_tiny_patch16_224'} }" +
            f"or compare variants. You wrote: {config.model.model_name}"
        )