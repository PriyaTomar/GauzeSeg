import logging
from functools import partial
import markdown
import pytorch_lightning as pl
from datetime import datetime
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
from pytorch_lightning import loggers as pl_loggers
from seg2ml.utils.yamls import ParameterCombination
import wandb

# make all tqdm pbars dynamic to fit any window resolution
tqdm = partial(tqdm, dynamic_ncols=True)

logging.basicConfig()
logger = logging.getLogger(__name__)


def log_text(key: str, log_string: str, trainer: pl.Trainer) -> None:
    if not log_string.startswith("\t"):
        log_string = fix_formatting(log_string)
    if isinstance(trainer.logger, pl.loggers.TensorBoardLogger):
        log_text_tensorboard(key, log_string, trainer)
    elif isinstance(trainer.logger, pl.loggers.WandbLogger):
        log_text_wandb(key, log_string)
    else:
        logger.error(f"pl.Trainer.logger of type {type(trainer.logger)} can not store text.")


def fix_formatting(log_string: str) -> str:
    """
    In markdown, we create a code block by indenting each line with a tab \t.
    Code blocks have fixed formatting, i.e. each character has the same width and
    spacing is kept.
    This makes sure that texts like classification reports are printed as they appear
    in the console.

    Parameters
    ----------
    log_string

    Returns
    -------
    formatted log_string
    """
    lines = log_string.split("\n")
    lines = ["\t" + line for line in lines]
    return "\n".join(lines)


def log_text_tensorboard(key: str, log_string: str, trainer: pl.Trainer):
    trainer.logger.experiment.add_text(key, log_string, global_step=trainer.current_epoch)


def log_text_wandb(key: str, log_string: str):
    try:
        wandb.log({key: wandb.Html(markdown_to_html(log_string))})
    except wandb.errors.Error as e:
        logger.warning(
            f"Unable to log string with wandb. "
            f"If this happens in the validation sanity check, you can ignore this message. "
            f'Error: "{str(e)}"'
        )
        return


def markdown_to_html(markdown_string: str) -> str:
    return markdown.markdown(markdown_string)

def get_datetime_str():
    return str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
def get_log_string(
    index: int,
    total: int,
    file_name: str,
    parameter_combination: Optional[ParameterCombination] = None,
) -> str:
    log_string = (
        f"\n"
        f"\t = = = = = = = = = = = = = = = = = = \n"
        f"\t RUNNING CONFIG NO. {index + 1} of {total} from {file_name}\n"
    )
    if parameter_combination is not None:
        log_string += "\t PARAMETERS:\n"
        for name, value in parameter_combination:
            log_string += f"\t\t {name}: {value}\n"
    return log_string
