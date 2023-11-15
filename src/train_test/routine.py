import torch

from abc import ABC, abstractmethod
from typing import Optional, Deque
from collections import deque

from src.utils.config_parser import TrainTest as TrainTestConfig
from src.models.model import Model
from lib.glass_defect_dataset.src.datasets.dataset import DatasetWrapper
from lib.glass_defect_dataset.src.utils.tools import Logger
from lib.glass_defect_dataset.config.consts import General as _CG


class TrainTest(ABC):

    train_str, val_str, test_str = _CG.DEFAULT_SUBSETS
    
    def __init__(self, train_test_config: TrainTestConfig, model: Model, dataset_wrapper: DatasetWrapper):
        self.train_test_config = train_test_config
        self.model = model
        self.dataset_wrapper = dataset_wrapper

        # CHECK to avoid linter's error change abstract property of DatasetWrapper.{split}_dataset to DatasetLauncher
        self.train_info: Optional[dict] = self.dataset_wrapper.train_dataset.info_dict
        self.val_info: Optional[dict] = self.dataset_wrapper.val_dataset.info_dict if self.dataset_wrapper.val_dataset is not None else None
        self.test_info: Optional[dict] = self.dataset_wrapper.test_dataset.info_dict

        self._model_config = self.model.config.model
        self.acc_var: Deque[float] = deque(maxlen=10)

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def test(self, model_path: str):
        ...

    def check_stop_conditions(self, loss: float, curr_acc: float, limit: float = 0.985, eps: float = 0.001) -> bool:
        if torch.isnan(torch.tensor(loss)).item():
            Logger.instance().error(f"Raised stop conditions because loss is NaN")
            return True

        if curr_acc < limit:
            return False
        
        if not len(self.acc_var) == self.acc_var.maxlen:
            self.acc_var.append(curr_acc)
            return False
        
        self.acc_var.popleft()
        self.acc_var.append(curr_acc)

        acc_var = torch.Tensor(list(self.acc_var))
        
        if torch.max(acc_var) > 0.999:
            Logger.instance().warning(f"Accuracy is 1.0: hit stop conditions")
            return True
        
        if torch.max(acc_var) - torch.min(acc_var) > 2 * eps:
            return False
        
        Logger.instance().warning(f"Raised stop condition: last {len(self.acc_var)} increment below {2 * eps}")
        return True


class TrainTestExample(TrainTest):

    def __init__(self, train_test_config: TrainTestConfig, model: Model, dataset: DatasetWrapper):
        super().__init__(train_test_config, model, dataset)

    def train(self):
        Logger.instance().debug("train example")

    def test(self, model_path: str):
        Logger.instance().debug("test example")