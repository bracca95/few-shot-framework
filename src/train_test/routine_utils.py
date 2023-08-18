from src.utils.tools import Logger
from src.models.model import Model
from src.train_test.routine import TrainTest, TrainTestExample
from src.train_test.standard_routine import StandardRoutine
from src.train_test.proto_routine import ProtoRoutine
from src.utils.config_parser import TrainTest as TrainTestConfig
from lib.glass_defect_dataset.src.datasets.dataset import CustomDataset

class RoutineBuilder:

    @staticmethod
    def build_routine(train_test_config: TrainTestConfig, model: Model, dataset: CustomDataset, debug: bool=False) -> TrainTest:
        if debug:
            Logger.instance().warning("DEBUGGING train/test routine started. Check if build_routine has `debug`=True")
            return TrainTestExample(train_test_config, model, dataset)

        if "compare" in model.config.model.model_name:
            Logger.instance().info("Standard train/test routine started.")
            return StandardRoutine(train_test_config, model, dataset)
        else:
            Logger.instance().info("Proto train/test routine started.")
            return ProtoRoutine(train_test_config, model, dataset)
