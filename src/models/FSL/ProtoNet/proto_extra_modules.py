import os
import torch

from typing import Optional, List

from src.models.model import Model
from src.models.LSTM.bilstm import BiLSTM
from src.models.MLP.proj_heads import ProjHead
from src.models.FSL.ProtoNet.distance_module import DistScale
from lib.glass_defect_dataset.src.utils.tools import Logger
from lib.glass_defect_dataset.config.consts import General as _CG

class ProtoEnhancements:

    ENH_IPN = "ipn"
    ENH_DIST = "dist_of_dists"
    ENH_APN = "apn"
    ENH_LSTM = "lstm"
    ENH_CONTR_LSTM = "contrastive_lstm"
    ENH_AUTOCORR = "autocorr"

    MOD_DISTSCALE = DistScale.__name__
    MOD_BILSTM = BiLSTM.__name__
    MOD_AUTOCORR_S = f"{ProjHead.__name__}Support"
    MOD_AUTOCORR_Q = f"{ProjHead.__name__}Query"

    def __init__(self, base_model: Model):
        self.base_model = base_model
        self.name = base_model.config.model.fsl.enhancement

        # a single enhancement may contain more than one module
        self.extra_modules = self.__init_modules()

    def __init_modules(self) -> dict:
        """Private method to init extra modules

        Load all the modules that an enhancement may need. If an enhancement does not require an additional module,
        append None

        Returns:
            dict of 
        """

        extra_modules = {  }
        config = self.base_model.config
        config_fsl = self.base_model.config.model.fsl

        if self.name == self.ENH_IPN:
            if not config_fsl.train_n_way == config_fsl.test_n_way:
                raise ValueError(
                    f"For distance scaling train/test n_way must be equal " +
                    f"({config_fsl.train_n_way} != {config_fsl.test_n_way})"
                )
            n_way = config_fsl.train_n_way
            module_dist_scale = DistScale(n_way * n_way, n_way)
            extra_modules[self.MOD_DISTSCALE] = module_dist_scale
        
        if self.name == self.ENH_LSTM or self.name == self.ENH_CONTR_LSTM:
            extr_out_size = self.base_model.get_out_size(1)
            module_lstm = BiLSTM(config, extr_out_size, extr_out_size // 2, config_fsl.train_k_shot_s)
            extra_modules[self.MOD_BILSTM] = module_lstm

        if self.name == self.ENH_AUTOCORR:
            extr_out_size = self.base_model.get_out_size(1)
            module_proj_support = ProjHead(config, extr_out_size * config_fsl.train_k_shot_s, extr_out_size // 2)
            module_proj_query = ProjHead(config, extr_out_size, extr_out_size // 2)
            extra_modules[self.MOD_AUTOCORR_S] = module_proj_support
            extra_modules[self.MOD_AUTOCORR_Q] = module_proj_query

        extra_modules = { k: v.to(_CG.DEVICE) for k, v in extra_modules.items() }
        return extra_modules

    def train(self):
        """Wrapper for torch.nn.Module.train()

        This methods wraps the train() settings for torch for the main model and, if needed, for the additional modules
        """

        self.base_model.train()

        if len(self.extra_modules) == 0:
            Logger.instance().debug(f"No extra module added in train. Enhance is {self.name}")
            return

        for module_name, module in self.extra_modules.items():
            module.train()
            Logger.instance().debug(f"Added extra module {module_name} in train")

    def eval(self):
        """Wrapper for torch.nn.Module.eval()

        This methods wraps the eval() settings for torch for the main model and, if needed, for the additional modules
        """

        self.base_model.eval()

        if len(self.extra_modules) == 0:
            Logger.instance().debug(f"No extra module added in test. Enhance is {self.name}")
            return
            
        for module_name, module in self.extra_modules.items():
            module.eval()
            Logger.instance().debug(f"Added extra module {module_name} in test")

    def save_models(self, model_path: str):
        """Save every torch model

        Wraps the torch.save() for the main model and, if needed, for every module. Provide the main model path (full)
        and the name of the other modules will be inferred automatically.

        Args:
            model_path (str): the main model path
        """

        torch.save(self.base_model.state_dict(), model_path)
        
        for module_name, module in self.extra_modules.items():
            filename, ext = os.path.basename(model_path).split(".")
            module_path = os.path.join(os.path.dirname(model_path), f"{filename}_{module_name}.{ext}")
            torch.save(module.state_dict(), module_path)

    def load_models(self, model_path: str):
        """Load every torch model

        Wraps the torch.load() for the main model and, if needed, for every module. Provide the main model path (full)
        and the name of the other modules will be inferred automatically.

        Args:
            model_path (str): the main model path
        """

        self.base_model.load_state_dict(torch.load(model_path))

        for module_name, module in self.extra_modules.items():
            filename, ext = os.path.basename(model_path).split(".")
            module_path = os.path.join(os.path.dirname(model_path), f"{filename}_{module_name}.{ext}")
            module.load_state_dict(torch.load(module_path))