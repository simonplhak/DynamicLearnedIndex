from __future__ import annotations

import functools
import subprocess
import time
from typing import TYPE_CHECKING, Callable, ParamSpec, TypeVar

import h5py
import psutil
import torch
from loguru import logger
from torch import Tensor

if TYPE_CHECKING:
    from torch.nn import Sequential

    from config import DatasetConfig


Param = ParamSpec('Param')
ReturnType = TypeVar('ReturnType')


def sizeof_fmt(num: float, suffix: str = 'B') -> str:
    # https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:  # noqa: PLR2004
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


def measure_runtime(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    @functools.wraps(func)
    def wrapper_measure_runtime(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
        start = time.time()
        result = func(*args, **kwargs)
        stop = time.time()

        logger.debug(f'Execution of {func.__name__} took {stop - start:.5}s.')

        return result

    return wrapper_measure_runtime


def measure_memory_usage(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    @functools.wraps(func)
    def wrapper_measure_memory_usage(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
        process = psutil.Process()
        start = process.memory_info().rss
        result = func(*args, **kwargs)
        stop = process.memory_info().rss
        logger.debug(f'Function {func.__name__} allocated {sizeof_fmt(stop - start)}.')
        return result

    return wrapper_measure_memory_usage


@measure_runtime
def load_data(config: DatasetConfig) -> tuple[Tensor, Tensor, Tensor]:
    X = torch.from_numpy(h5py.File(config.X, 'r')['emb'][: config.dataset_size])  # type: ignore
    Q = torch.from_numpy(h5py.File(config.Q, 'r')['emb'][:])  # type: ignore
    GT = torch.from_numpy(h5py.File(config.GT, 'r')['knns'][:])  # type: ignore

    assert X.dtype == torch.float16
    assert Q.dtype == torch.float32
    assert GT.dtype == torch.int32

    return X, Q, GT


def obtain_commit_hash() -> str:
    return subprocess.check_output(['git', 'describe', '--always']).strip().decode()  # noqa: S603, S607


def obtain_dirty_state() -> bool:
    return subprocess.call(['git', 'diff', '--quiet']) != 0  # noqa: S603, S607


def get_model_size(model: Sequential) -> int:
    """Calculate the total size in bytes of a PyTorch Sequential model.

    Includes both parameters and buffers.

    Args:
        model: PyTorch Sequential model to analyze

    Returns:
        Total size of model in bytes

    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())

    return param_size + buffer_size
