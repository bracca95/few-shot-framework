# inspired by quicktype.io

from __future__ import annotations

import os
import sys
import json

from functools import reduce
from dataclasses import dataclass, field
from typing import Optional, Union, List, Any, Callable, Iterable, Type, cast

from src.utils.tools import Tools, Logger
from config.consts import ConfigConst as _CC
from config.consts import ModelConfig as _CM
from config.consts import FSLConsts as _CFSL
from config.consts import TrainTestConfig as _CTT
from lib.glass_defect_dataset.config.consts import T
from lib.glass_defect_dataset.config.consts import General as _CG
from lib.glass_defect_dataset.src.utils.config_parser import DatasetConfig

def from_bool(x: Any) -> bool:
    Tools.check_instance(x, bool)
    return x

def from_int(x: Any) -> int:
    Tools.check_instance(x, int)
    return x

def from_float(x: Any) -> float:
    Tools.check_instance(x, float)
    return x

def from_str(x: Any) -> str:
    Tools.check_instance(x, str)
    return x

def from_none(x: Any) -> Any:
    Tools.check_instance(x, None)
    return x

def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    Tools.check_instance(x, list)
    return [f(y) for y in x]

def from_union(fs: Iterable[Any], x: Any):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    raise TypeError(f"{x} should be one out of {[type(f.__name__) for f in fs]}")


def to_class(c: Type[T], x: Any) -> dict:
    Tools.check_instance(x, c)
    return cast(Any, x).serialize()


@dataclass
class Fsl:
    episodes: int = _CG.DEFAULT_INT
    train_n_way: int = _CG.DEFAULT_INT
    train_k_shot_s: int = _CG.DEFAULT_INT
    train_k_shot_q: int = _CG.DEFAULT_INT
    test_n_way: int = _CG.DEFAULT_INT
    test_k_shot_s: int = _CG.DEFAULT_INT
    test_k_shot_q: int = _CG.DEFAULT_INT
    enhancement: Optional[str] = None

    @classmethod
    def deserialize(cls, obj: Any) -> Fsl:
        
        try:
            Tools.check_instance(obj, dict)
            episodes = from_int(obj.get(_CFSL.FSL_EPISODES))
            train_n_way = from_int(obj.get(_CFSL.FSL_TRAIN_N_WAY))
            train_k_shot_s = from_int(obj.get(_CFSL.FSL_TRAIN_K_SHOT_S))
            train_k_shot_q = from_int(obj.get(_CFSL.FSL_TRAIN_K_SHOT_Q))
            test_n_way = from_int(obj.get(_CFSL.FSL_TEST_N_WAY))
            test_k_shot_s = from_int(obj.get(_CFSL.FSL_TEST_K_SHOT_S))
            test_k_shot_q = from_int(obj.get(_CFSL.FSL_TEST_K_SHOT_Q))
            enhancement = from_union([from_none, from_str], obj.get(_CFSL.FSL_ENHANCEMENT))
        except TypeError as te:
            Logger.instance().error(te.args)
            sys.exit(-1)

        Logger.instance().info(
            f"episodes: {episodes}, " +
            f"train_n_way: {train_n_way}, train_k_shot_s: {train_k_shot_s}, train_k_shot_q: {train_k_shot_q}, " +
            f"test_n_way: {test_n_way}, test_k_shot_s: {test_k_shot_s}, test_k_shot_q: {test_k_shot_q}, " +
            f"enhancement: {enhancement}"
        )
        return Fsl(episodes, train_n_way, train_k_shot_s, train_k_shot_q, test_n_way, test_k_shot_s, test_k_shot_q, enhancement)

    def serialize(self) -> dict:
        result: dict = {}
        
        result[_CFSL.FSL_EPISODES] = from_int(self.episodes)
        result[_CFSL.FSL_TRAIN_N_WAY] = from_int(self.train_n_way)
        result[_CFSL.FSL_TRAIN_K_SHOT_S] = from_int(self.train_k_shot_s)
        result[_CFSL.FSL_TRAIN_K_SHOT_Q] = from_int(self.train_k_shot_q)
        result[_CFSL.FSL_TEST_N_WAY] = from_int(self.test_n_way)
        result[_CFSL.FSL_TEST_K_SHOT_S] = from_int(self.test_k_shot_s)
        result[_CFSL.FSL_TEST_K_SHOT_Q] = from_int(self.test_k_shot_q)
        result[_CFSL.FSL_ENHANCEMENT] = from_union([from_none, from_str], self.enhancement)

        Logger.instance().info(f"FSL serialized: {result}")
        return result


@dataclass
class Model:
    model_name: str = _CG.DEFAULT_STR
    fsl: Optional[Fsl] = None

    @classmethod
    def deserialize(cls, obj: Any) -> Model:
        try:
            Tools.check_instance(obj, dict)
            model_name = from_str(obj.get(_CM.CONFIG_MODEL_NAME))
            fsl = from_union([from_none, Fsl.deserialize], obj.get(_CM.CONFIG_FSL))
        except TypeError as te:
            Logger.instance().error(te.args)
            sys.exit(-1)

        Logger.instance().info(f"model_name: {model_name}, fsl: {fsl}")
        return Model(model_name, fsl)
    
    def serialize(self) -> dict:
        result: dict = {}

        result[_CM.CONFIG_MODEL_NAME] = from_str(self.model_name)
        result[_CM.CONFIG_FSL] = from_union([lambda x: to_class(Fsl, x), from_none], self.fsl)

        Logger.instance().info(f"Model serialized {result}")
        return result
    

@dataclass
class TrainTest:
    epochs: int = _CG.DEFAULT_INT
    batch_size: int = _CG.DEFAULT_INT
    model_test_path: Optional[str] = None
    learning_rate: Optional[float] = None
    optimizer: Optional[str] = None

    @classmethod
    def deserialize(cls, obj: Any) -> TrainTest:
        try:
            Tools.check_instance(obj, dict)
            epochs = abs(from_int(obj.get(_CTT.CONFIG_EPOCHS)))
            batch_size = abs(from_int(obj.get(_CTT.CONFIG_BATCH_SIZE)))
            model_test_path = from_union([from_none, from_str], obj.get(_CTT.CONFIG_MODEL_TEST_PATH))
            learning_rate = from_union([from_none, from_float], obj.get(_CTT.CONFIG_LEARNING_RATE))
            optimizer = from_union([from_none, from_str], obj.get(_CTT.CONFIG_OPTIMIZER))
        except TypeError as te:
            Logger.instance().error(te.args)
            sys.exit(-1)

        if model_test_path is not None:
            try:
                model_test_path = Tools.validate_path(model_test_path)
            except FileNotFoundError as fnf:
                msg = f"Check the test path again"
                Logger.instance().critical(f"{fnf.args}.\n{msg}")
                sys.exit(-1)

        Logger.instance().info(
            f"epochs: {epochs}, batch_size: {batch_size}, model_test_path: {model_test_path}, " +
            f"learning_rate: {learning_rate}, optimizer: {optimizer}"
        )
        
        return TrainTest(epochs, batch_size, model_test_path, learning_rate, optimizer)
    
    def serialize(self) -> dict:
        result: dict = {}

        result[_CTT.CONFIG_EPOCHS] = from_int(self.epochs)
        result[_CTT.CONFIG_BATCH_SIZE] = from_int(self.batch_size)
        result[_CTT.CONFIG_MODEL_TEST_PATH] = from_union([from_none, from_str], self.model_test_path)
        result[_CTT.CONFIG_LEARNING_RATE] = from_union([from_none, from_float], self.learning_rate)
        result[_CTT.CONFIG_OPTIMIZER] = from_union([from_none, from_str], self.optimizer)

        Logger.instance().info(f"TrainTest serialized: {result}")
        return result


@dataclass
class Config:
    experiment_name: str = "random_generated_name"
    dataset: DatasetConfig = DatasetConfig()
    model: Model = Model()
    train_test: TrainTest = TrainTest()

    @classmethod
    def deserialize(cls, obj: Any) -> Config:
        try:
            Tools.check_instance(obj, dict)
            experiment_name = from_str(obj.get(_CC.CONFIG_EXPERIMENT_NAME))
            dataset = DatasetConfig.deserialize(obj.get(_CC.CONFIG_DATASET))
            model = Model.deserialize(obj.get(_CC.CONFIG_MODEL))
            train_test = TrainTest.deserialize(obj.get(_CC.CONFIG_TRAIN_TEST))
        except TypeError as te:
            Logger.instance().critical(te.args)
            sys.exit(-1)
        
        Logger.instance().info(
            f"Config deserialized: experiment_name: {experiment_name}, dataset: {dataset}, model: {model}, " +
            f"train_test: {train_test}"
        )
        
        return Config(experiment_name, dataset, model, train_test)
    
    def serialize(self) -> dict:
        result: dict = {}
        
        # if you do not want to write null values, add a field to result if and only if self.field is not None
        result[_CC.CONFIG_EXPERIMENT_NAME] = from_str(self.experiment_name)
        result[_CC.CONFIG_DATASET] = to_class(DatasetConfig, self.dataset)
        result[_CC.CONFIG_MODEL] = to_class(Model, self.model)
        result[_CC.CONFIG_TRAIN_TEST] = to_class(TrainTest, self.train_test)

        Logger.instance().info(f"Config serialized {result}")
        return result


def read_from_json(str_path: str) -> Config:
    obj = Tools.read_json(str_path)
    return Config.deserialize(obj)

def write_to_json(config: Config, directory: str, filename: str) -> None:
    dire = None
    
    try:
        dire = Tools.validate_path(directory)
    except FileNotFoundError as fnf:
        Logger.instance().critical(f"{fnf.args}")
        sys.exit(-1)

    serialized_config = config.serialize()
    with open(os.path.join(dire, filename), "w") as f:
        json_dict = json.dumps(serialized_config, indent=4)
        f.write(json_dict)
