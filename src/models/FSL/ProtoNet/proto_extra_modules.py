import os
import torch

from typing import Optional, List

from src.models.model import Model
from src.models.FSL.ProtoNet.distance_module import DistScale
from src.utils.tools import Logger
from src.utils.config_parser import Fsl as FslConfig
from lib.glass_defect_dataset.config.consts import General as _CG

class ProtoEnhancements:

    MOD_IPN = "ipn"
    MOD_DIST = "dist_of_dists"
    MOD_APN = "apn"

    def __init__(self, base_model: Model, fsl_config: FslConfig):
        self.base_model = base_model
        self.fsl_config = fsl_config
        self.name = fsl_config.enhancement

        # a single enhancement may contain more than one module
        self.module_list = self.__init_modules()

    def __init_modules(self) -> List[Optional[torch.nn.Module]]:
        """Private method to init extra modules

        Load all the modules that an enhancement may need. If an enhancement does not require an additional module,
        append None

        Returns:
            List[Optional[torch.nn.Module]]
        """

        module_list = []

        if self.name == self.MOD_IPN:
            if not self.fsl_config.train_n_way == self.fsl_config.test_n_way:
                raise ValueError(
                    f"For distance scaling train/test n_way must be equal " +
                    f"({self.fsl_config.train_n_way} != {self.fsl_config.test_n_way})"
                )
            n_way = self.fsl_config.train_n_way
            module_list.append(DistScale(n_way * n_way, n_way).to(_CG.DEVICE))
        else:
            module_list.append(None)

        return module_list

    def train(self):
        """Wrapper for torch.nn.Module.train()

        This methods wraps the train() settings for torch for the main model and, if needed, for the additional modules
        """

        self.base_model.train()

        for module in self.module_list:
            if module is None:
                Logger.instance().debug(f"No extra module added in train. Enhance is {self.name}")
                return
        
            module.train()
            Logger.instance().debug(f"Added extra module {module.__class__.__name__} in train")

    def eval(self):
        """Wrapper for torch.nn.Module.eval()

        This methods wraps the eval() settings for torch for the main model and, if needed, for the additional modules
        """

        self.base_model.eval()

        for module in self.module_list:
            if module is None:
                Logger.instance().debug(f"No extra module added in test. Enhance is {self.name}")
                return
            
            module.eval()
            Logger.instance().debug(f"Added extra module {module.__class__.__name__} in test")

    def save_models(self, model_path: str):
        """Save every torch model

        Wraps the torch.save() for the main model and, if needed, for every module. Provide the main model path (full)
        and the name of the other modules will be inferred automatically.

        Args:
            model_path (str): the main model path
        """

        torch.save(self.base_model.state_dict(), model_path)
        
        for module in self.module_list:
            if module is not None:
                filename, ext = os.path.basename(model_path).split(".")
                module_path = os.path.join(os.path.dirname(model_path), f"{filename}_{module.__class__.__name__}.{ext}")
                torch.save(module.state_dict(), module_path)

    def load_models(self, model_path: str):
        """Load every torch model

        Wraps the torch.load() for the main model and, if needed, for every module. Provide the main model path (full)
        and the name of the other modules will be inferred automatically.

        Args:
            model_path (str): the main model path
        """

        self.base_model.load_state_dict(torch.load(model_path))

        for module in self.module_list:
            if module is not None:
                filename, ext = os.path.basename(model_path).split(".")
                module_path = os.path.join(os.path.dirname(model_path), f"{filename}_{module.__class__.__name__}.{ext}")
                module.load_state_dict(torch.load(module_path))