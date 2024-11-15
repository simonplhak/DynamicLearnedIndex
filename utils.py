from __future__ import annotations

import functools
import subprocess
import time
from typing import TYPE_CHECKING, Any, Callable

import h5py
import psutil
import torch
from loguru import logger
from torch import Tensor

if TYPE_CHECKING:
    from configuration import DatasetConfig


def sizeof_fmt(num: float, suffix: str = 'B') -> str:
    # https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:  # noqa: PLR2004
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


def measure_runtime(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper_measure_runtime(*args, **kwargs) -> Any:  # noqa: ANN401, ANN002, ANN003
        start = time.time()
        result = func(*args, **kwargs)
        stop = time.time()

        logger.info(f'Execution of {func.__name__} took {stop - start:.5}s.')

        return result

    return wrapper_measure_runtime


def measure_memory_usage(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper_measure_memory_usage(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        process = psutil.Process()
        start = process.memory_info().rss
        result = func(*args, **kwargs)
        stop = process.memory_info().rss
        # ? What if some objects are garbage collected during the function's execution?
        logger.debug(f'Function {func.__name__} allocated {sizeof_fmt(stop - start)}.')
        return result

    return wrapper_measure_memory_usage


@measure_runtime
def load_data(config: DatasetConfig) -> tuple[Tensor, Tensor, Tensor]:
    X = torch.from_numpy(h5py.File(config.X, 'r')['emb'][: config.dataset_size]).to(torch.float32)  # type: ignore
    Q = torch.from_numpy(h5py.File(config.Q, 'r')['emb'][:]).to(torch.float32)  # type: ignore
    GT = torch.from_numpy(h5py.File(config.GT, 'r')['knns'][:])  # type: ignore
    return X, Q, GT


def obtain_commit_hash() -> str:
    return subprocess.check_output(['git', 'describe', '--always']).strip().decode()  # noqa: S603, S607


def obtain_dirty_state() -> bool:
    return subprocess.call(['git', 'diff', '--quiet']) != 0  # noqa: S603, S607
