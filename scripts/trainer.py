#import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 


import argparse
import copy
import logging
# import coloredlogs

from typing import Any, Dict, List, Optional, Tuple
import os
import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from numpy.random import RandomState, SeedSequence, MT19937
import random
from seg2ml.utils.logging import get_log_string, get_datetime_str
from seg2ml.data.gauze.loader import GauzeDataModule
from seg2ml.utils.helper import create_class_instance, create_instance, timestamp, load_model_from_path
from seg2ml.utils.yamls import (
    ParameterCombination,
    expand_params,
    load_params,
    write_params,
)
import warnings 
warnings.filterwarnings("ignore")

# needed for reproducable result in gridsearch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
rs = RandomState(MT19937(SeedSequence(0)))
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(0) 
pl.seed_everything(42, workers=True)


#coloredlogs.install(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)


def init_deterministic_algorithms(val = True):
    print("deterministoc val", val)
    torch.use_deterministic_algorithms(val) # Should be true but one function does not have a determinsitic implementation!
    is_deterministic = torch.are_deterministic_algorithms_enabled()

    if is_deterministic:
        print("Deterministic algorithms are enabled.")
    else:
        print("Deterministic algorithms are disabled.")


def init_datamodule(dataset_params: Dict = {}, dataloader_params: Dict = {}):
    datamodule =GauzeDataModule(**dataset_params,
                                  **dataloader_params
                                  )
    return datamodule

def init_model(model_params: Dict, checkpoint=None, checkpoint_config=None):
    logger.info(f"Initializing model")
    ckpt_path = model_params["args"].get("ckpt_path", None)
    #ckpt_path = None
    if ckpt_path:
        model = load_model_from_path(model_params["module"], model_params["name"], model_params["args"], ckpt_path)
        print("Model ckpt path", ckpt_path)
    else: 
        model = create_class_instance(model_params["module"], model_params["name"], model_params["args"]
    )
    return model

def init_callbacks(params):
    callbacks = []
    if params.get("callbacks") is not None:
        for callback_params in params["callbacks"]:
            callback = create_class_instance(
                    callback_params["module"],
                    callback_params["name"],
                    callback_params["args"])
            callbacks.append(callback)

    save_checkpoints = params["model_checkpoints"].get("save_checkpoints", False)
    checkpoints_dir = os.path.join(params["model_checkpoints"]["checkpoints_dir"], params["run_name"], params["run_time"])
    if save_checkpoints:
        os.makedirs(checkpoints_dir, exist_ok=True)
    write_params(params, os.path.join(checkpoints_dir, "config.yaml"))

    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor=params["model_checkpoints"].get("best_model_metric", "valid/Dice"),
        save_last=False,  # only save last if any checkpoints are saved
        verbose=True,
        mode='max', #params["model_checkpoints"].get("mode"),
        save_top_k=1#
    )
    callbacks.append(model_checkpoint)
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    model_summary = ModelSummary(max_depth=3)
    callbacks.append(model_summary)
    return callbacks

def train_params(params):
    
    if "run" in params:
        run_name = params["run"]
    elif "name" in params:
        run_name = params["name"]
    else:
        run_name = "no_run_name"
    params["run_time"] = timestamp()
    #params["run_name"] = run_name
    #setting run name to timestamp
    params["run_name"] = run_name + "__" + get_datetime_str()
    
    data_module = init_datamodule(params["dataset"], params["dataloader"])
    model = init_model(params["model"])
    
    hparams = {"model": params["model"],
               "dataloader": params["dataloader"]
                # "optimizer": type(optimizer).__name__,
                # "scheduler": lr_scheduler,
                }
    
    callbacks = init_callbacks(params)
    
    loggers = []
    tensorboard_logger = TensorBoardLogger(save_dir=params["logging"]["save_dir"], name=get_datetime_str())
    tensorboard_logger.log_hyperparams(hparams)
    loggers.append(tensorboard_logger)
    
    if params["logging"]["wandb_logging"]:
        wandb_logger = WandbLogger(project=params["logging"]["wandb_project_name"], name=params["run_name"])
        wandb_logger.log_hyperparams(hparams)
        loggers.append(wandb_logger) 
    
    trainer = pl.Trainer(logger=loggers, 
                         callbacks=callbacks,
                         default_root_dir=params["model_checkpoints"]["save_checkpoints"], # saves checkpoints to 'some/path/' at every epoch end
                         **params["trainer"]
                        )
    
    #trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
    trainer.fit(model, datamodule=data_module)
    trainer.test(dataloaders={"test": data_module.test_dataloader(), "test2": data_module.test2_dataloader()}, ckpt_path='best')
    wandb.join()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        nargs="+",
        help="Paths to config.yaml files. "
        'Add multiple files for a gridseach by "-c /configs/path1.yaml /configs/path2.yaml"',
    )
    args = parser.parse_args()

    for path in args.config:
        assert os.path.exists(path), f"Config file '{path}' not found!"
    logger.info(f"Loading params from {args.config}")
    gs_params, parameter_combinations, file_names = [], [], []
    for config_path in args.config:
        _gs_params, _parameter_combinations = expand_params(load_params(config_path), adjust_run_name=False)
        gs_params.extend(_gs_params)
        parameter_combinations.extend(_parameter_combinations)
        file_names.extend([config_path] * len(_gs_params))

    logger.info(f"Found {len(gs_params)} params for grid-search.")
    
    for index, (params, parameter_combination, file_name) in enumerate(
        zip(gs_params, parameter_combinations, file_names)
    ):
        if parameter_combination:
            gs_param = ["_"+parameter_combination[i][0].split('.')[-1] for i in range(len(parameter_combination))]
            gs_value = [str(parameter_combination[i][1]) for i in range(len(parameter_combination))]
            gs_name = [gs_param[i]+gs_value[i] for i in range(len(gs_value))]
        else:
            gs_name = None

        params = copy.deepcopy(params)
        if gs_name:
            for i in range(len(gs_name)):
                params["run"] = params["run"] + gs_name[i]

        logger.info(get_log_string(index, len(gs_params), file_name, parameter_combination))
        init_deterministic_algorithms(params.get("use_deterministic", True))
        train_params(params)
    
if __name__ == "__main__":
    main()
