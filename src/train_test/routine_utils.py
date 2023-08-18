from src.utils.tools import Logger
from src.models.model import Model
from src.train_test.routine import TrainTest, TrainTestExample
from src.train_test.standard_routine import StandardRoutine
from src.train_test.proto_routine import ProtoRoutine
from lib.glass_defect_dataset.src.datasets.dataset import CustomDataset

class RoutineBuilder:

    @staticmethod
    def build_routine(name: str, model: Model, dataset: CustomDataset, debug: bool=False) -> TrainTest:
        if debug:
            Logger.instance().warning("DEBUGGING train/test routine started. Check if build_routine has `debug`=True")
            return TrainTestExample(model, dataset)

        if "compare" in name:
            Logger.instance().info("Standard train/test routine started.")
            return StandardRoutine(model, dataset)
        else:
            Logger.instance().info("Proto train/test routine started.")
            return ProtoRoutine(model, dataset)
