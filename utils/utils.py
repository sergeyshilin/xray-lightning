from __future__ import print_function
from typing import Dict

import ast
import importlib
import pickle
import random

import numpy as np
import torch
import torch.distributed as dist

from collections import OrderedDict
from omegaconf import OmegaConf
from typing import Any


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Parameters
    ----------
    data : object
        Any picklable object

    Returns
    -------
    list[data]
        List of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(
            torch.empty((max_size,), dtype=torch.uint8, device="cuda")
        )
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def collate_fn(batch):
    """TODO Add missing docstring."""
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """TODO Add missing docstring."""

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def is_dist_avail_and_initialized():
    """TODO Add missing docstring."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """TODO Add missing docstring."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def flatten_omegaconf(d, sep="_"):
    """TODO Add missing docstring."""
    d = OmegaConf.to_container(d)
    obj = OrderedDict()

    def recurse(t, parent_key=""):

        if isinstance(t, list):
            for i in range(len(t)):
                recurse(
                    t[i], parent_key + sep + str(i) if parent_key else str(i)
                )
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)
    obj = {k: v for k, v in obj.items() if type(v) in [int, float]}

    return obj


# https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.

    Parameters
    ----------
    obj_path : str
        Path to an object to be extracted, including the object name.
    default_obj_path : str, optional
        Default object path., by default ""

    Returns
    -------
    Any
        Extracted object.

    Raises
    ------
    AttributeError
        When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = (
        obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    )
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
        )
    return getattr(module_obj, obj_name)


def read_labels(filepath: str = "") -> Dict[str, int]:
    """
    Read the file with labels into a dictionary.

    Parameters
    ----------
    filepath : str, optional
        File path, by default ""

    Returns
    -------
    Dict[str, int]
        Labels dictionary, e.g. {"cat": 0, "dog": 1}
    """
    labels = {}

    if filepath != "":
        with open(filepath, "r") as content:
            labels = ast.literal_eval(content.read())

    return labels


def set_seed(seed: int = 42) -> None:
    """
    Set random seed globally.

    Parameters
    ----------
    seed : int, optional
        Random seed, by default 42
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
