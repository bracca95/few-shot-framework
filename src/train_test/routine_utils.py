from torch import nn

from src.train_test.routine import TrainTest
from src.train_test.standard_routine import StandardRoutine
from src.train_test.proto_routine import ProtoRoutine
from src.datasets.defectviews import DefectViews
from config.consts import SubsetsDict

class RoutineBuilder:

    @staticmethod
    def build_routine(name: str, model: nn.Module, dataset: DefectViews, subsets_dict: SubsetsDict) -> TrainTest:
        if "compare" in name:
            return StandardRoutine(model, dataset, subsets_dict)
        else:
            return ProtoRoutine(model, dataset, subsets_dict)