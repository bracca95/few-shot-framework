import torch
from torch import nn
from typing import Optional, Union, Tuple

from src.utils.tools import Logger
from src.utils.config_parser import Config
from lib.glass_defect_dataset.src.datasets.dataset import CustomDataset

class Model(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        Logger.instance().info(f"Model instantiated: {self.__class__.__name__}")

    def forward(self, x):
        pass

    @staticmethod
    def get_batch_size(config: Config) -> int:
        """Get the batch size to feed the model

        Batch size can be either set with the specific field in config.json or it is the result of other computations,
        depending on the method (e.g. protonet has a batch size that depends on N-way-K-shot).

        Args:
            config (Config)

        Returns:
            batch size (int)
        """

        if config.model.fsl is None or "compare" in config.model.model_name.lower():
            Logger.instance().debug(f"batch size is {config.train_test.batch_size}")
            return config.train_test.batch_size
        
        Logger.instance().debug(f"using protonet-like network to compute batch size")
        return config.model.fsl.train_n_way * config.model.fsl.train_k_shot_s + \
            config.model.fsl.test_n_way * config.model.fsl.test_k_shot_q

    def get_out_size(self, pos: Optional[int]) -> Union[torch.Size, int]:
        """Get the output size of a model
        
        This is useful both to know both for extractors or any other Module. The passed fake tensor is shaped according
        to the batch size.

        Args:
            pos (Optional[int]): if specified, returns the exact size in position `pos`; full torch.Size otherwise

        Returns:
            either an integer for the location specified by `pos` or torch.Size
        """
        
        batch_size = Model.get_batch_size(self.config)
        n_channels = len(self.config.dataset.dataset_mean) if self.config.dataset.dataset_mean is not None else 1
        x = torch.randn(batch_size, n_channels, self.config.dataset.image_size, self.config.dataset.image_size)
        
        with torch.no_grad():
            output = self.forward(x)

        # assuming a flat tensor so that shape = (batch_size, feature_vector, Opt[unpooled], Opt[unpooled])
        if pos is not None:
            if pos > len(output.shape):
                raise ValueError(f"required position {pos}, but model output size has {len(output.shape)} values.")
            return output.shape[pos]
        
        return output.shape