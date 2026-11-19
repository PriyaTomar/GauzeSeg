# coding: utf-8

import copy
import hashlib
import itertools
import json
import os
import shutil
from datetime import datetime as dt
from functools import reduce
from importlib import import_module
from logging import Logger
from typing import Dict, List, Tuple
#from seg2ml.models.lightning_model import LightningModel

import numpy as np
import torch
from scipy import linalg
import importlib

def get_model_capacity(model):
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,}")
            total += param.numel()
    print(f"\nTotal trainable parameters: {total:,}")


def timestamp(format: str = "%Y-%m-%d-%H-%M") -> str:
    return dt.now().strftime(format)


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def bytes_to_gigabytes(x: int) -> float:
    return np.round(x / (1024 ** 3), 2)


def get_file_size(file_path: str) -> float:
    assert os.path.isfile(file_path)
    return bytes_to_gigabytes(os.path.getsize(file_path))


def get_disk_usage(path: str) -> Tuple[str, float, float, float]:
    while not os.path.isdir(path):
        path = os.path.dirname(path)
    total, used, free = shutil.disk_usage(path)
    return (
        path,
        bytes_to_gigabytes(total),
        bytes_to_gigabytes(used),
        bytes_to_gigabytes(free),
    )


def hash_string(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def hash_dict(x: Dict) -> str:
    return hash_string(json.dumps(x, sort_keys=True))

def load_model_from_path(module_name, class_name, param_args, checkpoint_path):
    module = importlib.import_module(module_name)

    # Get the model class from the module
    model_class = getattr(module, class_name)
    # Load the model from checkpoint
    model = model_class.load_from_checkpoint(checkpoint_path, **param_args)
    return model


def create_class_instance(module_name, class_name, param_args: Dict = None, *args, **kwargs):
    """Create an instance of a given class.

    :param module_name: where the class is located
    :param class_name:
    :param param_args: arguments needed for the class constructor, as dict from param dict
    :param args: arguments needed for the class constructor, as additional positional arguments
    :param kwargs: arguments needed for the class constructor, as additional keyword arguments
    :returns: instance of 'class_name'

    """

    module = import_module(module_name)
    clazz = getattr(module, class_name)
    instance = clazz(*args, **(param_args or {}), **kwargs)
    return instance


def create_instance(name, param_args, *args, **kwargs):
    """Creates an instance of class given configuration.

    :param name: of the module we want to create
    :param params: dictionary containing information how to instantiate the class
    :returns: instance of a class
    :rtype:

    """
    i_params = param_args[name]
    if type(i_params) in [list, tuple]:
        instance = [create_class_instance(p["module"], p["name"], p.get("args", {}), *args, **kwargs) for p in i_params]
    else:
        instance = create_class_instance(
            i_params["module"],
            i_params["name"],
            i_params.get("args", {}),
            *args,
            **kwargs,
        )
    return instance


def create_nonlinearity(name):
    """
    Returns instance of non-linearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)
    instance = clazz()

    return instance


def get_class_nonlinearity(name):
    """
    Returns non-linearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)
    return clazz


def create_cost_function(name, *args):
    """
    Returns instance of cost functions (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)
    instance = clazz(*args)

    return instance


def to_one_hot(labels, num_classes):
    """
    Convert tensor of labels to one hot encoding of the labels.
    :param labels: to be encoded
    :param num_classes:
    :return:
    """
    shape = labels.size()
    shape = shape + (num_classes,)
    one_hot = torch.zeros(shape, dtype=torch.float, device=labels.device)
    dim = 1 if len(shape) == 2 else 2
    one_hot.scatter_(dim, labels.unsqueeze(-1), 1)
    return one_hot


def get_device(params: dict, rank: int = 0, logger: Logger = None, no_cuda: bool = False) -> torch.device:
    """

    :param params:
    :param logger:
    :return: returns the device
    """
    gpus = params.get("gpus", [])
    if not no_cuda and len(gpus) > 0:
        if not torch.cuda.is_available():
            if logger is not None:
                logger.warning("No GPU's available. Using CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(gpus[rank]))
    else:
        device = torch.device("cpu")
    return device


def is_primitive(v):
    """
    Checks if v is of primitive type.
    """
    return isinstance(v, (int, float, bool))


def is_array(v):
    return isinstance(v, (np.ndarray, torch.Tensor))


def free_params(module):
    if type(module) is not list:
        module = [module]
    for m in module:
        for p in m.parameters():
            p.requires_grad = True


def frozen_params(module):
    if type(module) is not list:
        module = [module]
    for m in module:
        for p in m.parameters():
            p.requires_grad = False


def sum_dictionaries(dicts: List):
    """
    Sums the values of the common keys in dictionary.

    Parameters
    ----------
    dicts (list) dictionaries containing numeric values

    Returns
    -------
    dictionary with summed values
    """

    def reducer(accumulator, element):
        for key, value in element.items():
            accumulator[key] = accumulator.get(key, 0) + value
        return accumulator

    return reduce(reducer, dicts, {})


def sample(dist, mode=None, unk_idx=None):
    """
    Auxiliary sampling method.
    """
    if mode in ["sample-no-unk", "greedy-no-unk"] and unk_idx is None:
        raise ValueError("Unknown index for the <unk> token!")
    if mode == "greedy":
        _, _sample = torch.topk(dist, 1, dim=-1)
    elif mode == "sample":
        sample_prob = torch.nn.functional.softmax(dist, dim=-1).squeeze(1)
        _sample = torch.multinomial(sample_prob, num_samples=1)
    elif mode == "sample-no-unk":
        # reduce chances for <unk>
        dist[:, :, unk_idx] = dist.min()
        sample_prob = torch.nn.functional.softmax(dist, dim=-1).squeeze(1)
        _sample = torch.multinomial(sample_prob, num_samples=1)
    elif mode == "greedy-no-unk":
        # prevent <unk>
        dist[:, :, unk_idx] = dist.min()
        _, _sample = torch.topk(dist, 1, dim=-1)
    else:
        raise ValueError(f"Unknown sampling mode = {mode}")

    _sample = _sample.squeeze()

    return _sample


def get_file_line_number(file_path: str) -> int:
    with open(file_path, "r", encoding="utf-8") as f:
        buf_size = 1024 ** 2
        read_f = f.read
        lines: int = 0
        buf = read_f(buf_size)
        while buf:
            lines += buf.count("\n")
            buf = read_f(buf_size)
        return lines


def gumbel_softmax():
    raise NotImplementedError
