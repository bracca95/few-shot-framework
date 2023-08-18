import os
import sys
import torch
import wandb
import random
import numpy as np

from src.models.model_utils import ModelBuilder
from src.train_test.routine_utils import RoutineBuilder
from lib.glass_defect_dataset.src.datasets.dataset_utils import DatasetBuilder
from src.utils.config_parser import Config, read_from_json, write_to_json
from src.utils.tools import Logger
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

if __name__=="__main__":
    try:
        config = read_from_json("config/config.json")
    except Exception as e:
        Logger.instance().critical(e.args)
        sys.exit(-1)

    try:
        dataset = DatasetBuilder.load_dataset(config.dataset)
    except ValueError as ve:
        Logger.instance().critical(ve.args)
        sys.exit(-1)

    # compute mean and variance of the dataset if not done yet
    if config.dataset.dataset_mean is None and config.dataset.dataset_std is None:
        Logger.instance().warning("No mean and std set: computing and storing values.")
        mean, std = dataset.compute_mean_std(dataset)
        config.dataset.dataset_mean = mean.tolist()
        config.dataset.dataset_std = std.tolist()
        write_to_json(config, os.getcwd(), "config/config.json")
        sys.exit(0)

    #### DEBUG - tested up to here ####
    ## start program
    wandb_mode = "disabled" if config.experiment_name == "disabled" else "online"
    wandb.init(
        mode=wandb_mode,
        project=config.experiment_name,
        config={
            "learning_rate": 0.001,
            "architecture": config.fsl.model,
            "dataset": config.dataset_type,
            "epochs": config.epochs,
        }
    )

    # instantiate model
    try:
        model = ModelBuilder.load_model(config, len(dataset.label_to_idx))
        model = model.to(_CG.DEVICE)
    except ValueError as ve:
        Logger.instance().critical(ve.args)
        sys.exit(-1)
    
    # split dataset
    subsets_dict = DatasetBuilder.split_dataset(dataset, config.dataset_splits)
    
    # train/test
    routine = RoutineBuilder.build_routine(config.fsl.model, model, dataset, subsets_dict)
    
    if config.fsl.model_test_path is None:
        routine.train(config)
        model_path = os.path.join(os.getcwd(), "output/best_model.pth")
    
    model_path = config.fsl.model_test_path if config.fsl.model_test_path is not None else model_path
    routine.test(config, model_path)

    wandb.save("log.log")
    wandb.finish()