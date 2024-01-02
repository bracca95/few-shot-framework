import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, Tensor
from torch.types import _dtype
from typing import Optional, Union, List
from sklearn.manifold import TSNE
from torchvision.transforms import transforms

from src.utils.config_parser import Config
from lib.glass_defect_dataset.src.utils.tools import Logger
from lib.glass_defect_dataset.config.consts import General as _CG


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
            output = self.forward(x.to(_CG.DEVICE))

        if type(output) is tuple:
            output = output[0]

        # assuming a flat tensor so that shape = (batch_size, feature_vector, Opt[unpooled], Opt[unpooled])
        if pos is not None:
            if pos > len(output.shape):
                raise ValueError(f"required position {pos}, but model output size has {len(output.shape)} values.")
            return output.shape[pos]
        
        return output.shape
    
    @staticmethod
    def one_hot_encoding(y: Tensor, n_classes: int, dtype: _dtype=torch.float):
        one = 1.0 if dtype is torch.float else int(1)
        one_hot_labels = torch.zeros((y.size(0), n_classes), dtype=dtype, device=_CG.DEVICE)
        one_hot_labels[range(len(y)), y] = one
        
        return one_hot_labels
    
    @staticmethod
    def plot_tsne(embeddings: torch.Tensor, n_classes: int, n_samples: int, epoch: Optional[int]=None):
        if not len(embeddings.shape) == 2:
            Logger.instance.warning(f"Failed t-sne: embeddings should have 2 dim, have {len(embeddings.shape)} instead")

        if not embeddings.device == torch.device("cpu"):
            embeddings = embeddings.detach().cpu()

        labels = np.repeat(np.arange(n_classes), n_samples)
        data_np = embeddings.numpy()

        # reduce the dimensionality to 2D
        perplex = 30 if n_classes * n_samples > 30 else 20
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplex)
        data_tsne = tsne.fit_transform(data_np)

        # plot t-SNE with colored points based on classes
        plt.figure(figsize=(8, 6))
        for class_label in range(n_classes):
            mask = (labels == class_label)
            plt.scatter(data_tsne[mask, 0], data_tsne[mask, 1], label=f"class {class_label}")

        ep = f"_epoch_{epoch}" if epoch is not None else ""
        filename = os.path.join(os.getcwd(), "output", f"tsne_epoch{ep}.png")
        plt.title("t-SNE Visualization with Class Colors")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def visual_attention(img: Tensor, att: Tensor, mean: Optional[List[float]], std: Optional[List[float]], ep: int):
        if att is None:
            return

        # prepare: select a random image, get its size and check if de-normalization is needed
        select: int = torch.randint(0, img.size(0), (1,)).item()
        s = img.size(-1)
        denorm = nn.Identity()
        if mean is not None and std is not None:
            denorm = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])

        # get one image and its corresponding attention
        att = att[select].detach().cpu()
        img = img[select].detach().cpu()
        img = denorm(img)

        # resize attention to match the input size and normalize
        attention_weights = nn.functional.interpolate(att.unsqueeze(0), size=(s, s), mode='bilinear', align_corners=False)
        attention_weights = attention_weights.squeeze()
        attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-6)
        heatmap = attention_weights.numpy()

        # plot with overlay
        plt.figure()
        plt.imshow(transforms.ToPILImage()(img))
        plt.imshow(heatmap, cmap='viridis', alpha=0.5)
        plt.title('Attention Heatmap')
        plt.savefig(os.path.join(os.getcwd(), "output", f"attention_{ep}.png"))
        plt.close()