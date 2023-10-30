import os
import sys
import torch
import wandb
import random
import numpy as np
import argparse

from src.models.model_utils import ModelBuilder, YoloModelBuilder
from src.train_test.routine_utils import RoutineBuilder
from src.train_test.proto_routine import ProtoInference
from src.utils.config_parser import Config, read_from_json, write_to_json
from src.utils.tools import Logger
from lib.glass_defect_dataset.src.datasets.dataset_utils import DatasetBuilder, YoloDatasetBuilder
from lib.glass_defect_dataset.src.datasets.dataset import CustomDataset
from lib.glass_defect_dataset.config.consts import General as _CG

SEED = 1234         # with the first protonet implementation I used 7

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


### TODOs ###
# online augmentation
#############

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", nargs="?", type=str, default=None)
args = vars(parser.parse_args())

def run_yolo(config: Config):
    dataset = YoloDatasetBuilder.load_dataset(config.dataset)
    model = YoloModelBuilder.load_model(config)

    model.execute(dataset)

def run_inference(config: Config):
    import copy
    support_set_config = copy.deepcopy(config.dataset)
    query_set_config = copy.deepcopy(config.dataset)

    # override support and query paths
    support_set_config.dataset_path = os.path.join(support_set_config.dataset_path, ProtoInference.SUPPORT)
    query_set_config.dataset_path = os.path.join(query_set_config.dataset_path, ProtoInference.QUERY)

    # load both dataset separately
    support_set = DatasetBuilder.load_dataset(support_set_config)
    query_set = DatasetBuilder.load_dataset(query_set_config)
    
    # load model
    model = ModelBuilder.load_model(config, len(support_set.label_to_idx.keys()))
    model = model.to(_CG.DEVICE)
    
    infer = ProtoInference(config, model, support_set, query_set)
    infer.test(config.train_test.model_test_path)

def main(config_path: str):
    try:
        config = read_from_json(config_path)
    except Exception as e:
        Logger.instance().critical(e.args)
        sys.exit(-1)

    # check if YOLO
    if "yolo" in config.dataset.dataset_type:
        run_yolo(config)
        Logger.instance().debug(f"YOLO ended its execution. Quitting...")
        sys.exit(0)

    # check if inference mode
    if "inference" in config.dataset.dataset_type:
        run_inference(config)
        Logger.instance().debug(f"Inference ended")
        sys.exit(0)

    try:
        dataset = DatasetBuilder.load_dataset(config.dataset)
        dataset.save_sample_image_batch(dataset, os.path.join(os.getcwd(), "output"))
    except ValueError as ve:
        Logger.instance().critical(ve.args)
        sys.exit(-1)

    # compute mean and variance of the dataset if not done yet
    if config.dataset.normalize and config.dataset.dataset_mean is None and config.dataset.dataset_std is None:
        Logger.instance().warning("No mean and std set: computing and storing values.")
        mean, std = dataset.compute_mean_std(dataset)
        config.dataset.dataset_mean = mean.tolist()
        config.dataset.dataset_std = std.tolist()
        write_to_json(config, os.getcwd(), config_path)
        
        # reload
        config = read_from_json(config_path)
        dataset = DatasetBuilder.load_dataset(config.dataset)

    ## start program
    wandb_mode = "disabled" if config.experiment_name == "disabled" else "online"
    wandb.init(
        mode=wandb_mode,
        project=config.experiment_name,
        config={
            "learning_rate": config.train_test.learning_rate,
            "architecture": config.model.model_name,
            "dataset": config.dataset.dataset_type,
            "epochs": config.train_test.epochs,
        }
    )

    # instantiate model
    try:
        model = ModelBuilder.load_model(config, len(dataset.label_to_idx))
        model = model.to(_CG.DEVICE)
    except ValueError as ve:
        Logger.instance().critical(ve.args)
        sys.exit(-1)
    
    # train/test
    routine = RoutineBuilder.build_routine(config.train_test, model, dataset)
    
    if config.train_test.model_test_path is None:
        routine.train()
        model_path = os.path.join(os.getcwd(), "output/best_model.pth")
    
    model_path = config.train_test.model_test_path if config.train_test.model_test_path is not None else model_path
    routine.test(model_path)

    wandb.save("output/log.log")
    wandb.finish()

if __name__=="__main__":
    config_file_path = args["config_file"] if args["config_file"] is not None else "config/config.json"
    Logger.instance().debug(f"config file located at {config_file_path}")
    main(config_file_path)