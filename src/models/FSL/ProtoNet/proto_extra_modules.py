from typing import Optional

from src.models.FSL.ProtoNet.distance_module import DistScale
from src.utils.tools import Logger
from src.utils.config_parser import Fsl as FslConfig
from lib.glass_defect_dataset.config.consts import General as _CG

class ProtoEnhancements:

    def __init__(self, fsl_config: FslConfig):
        self.fsl_config = fsl_config
        self.name = fsl_config.enhancement

        if self.name == "ipn":
            if not self.fsl_config.train_n_way == self.fsl_config.test_n_way:
                raise ValueError(f"For distance scaling train/test n_way must be equal \
                                 ({self.fsl_config.train_n_way} != {self.fsl_config.test_n_way})")
            n_way = self.fsl_config.train_n_way
            self.module = DistScale(n_way * n_way, n_way).to(_CG.DEVICE)
        else:
            self.module = None

    def train(self):
        if self.module is None:
            Logger.instance().debug(f"No extra module added in train. Enhance is {self.name}")
            return
        
        self.module.train()
        Logger.instance().debug(f"Added extra module {self.module.__class__.__name__} in train")

    def eval(self):
        if self.module is None:
            Logger.instance().debug(f"No extra module added in test. Enhance is {self.name}")
            return
        
        self.module.eval()
        Logger.instance().debug(f"Added extra module {self.module.__class__.__name__} in test")