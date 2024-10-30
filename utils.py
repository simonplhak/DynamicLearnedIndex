from __future__ import annotations

import functools
import subprocess
import time
from typing import TYPE_CHECKING, Any, Callable

import h5py
import torch
from loguru import logger
from torch import Tensor

if TYPE_CHECKING:
    from configuration import DatasetConfig


def measure_runtime(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper_measure_runtime(*args, **kwargs) -> Any:  # noqa: ANN401, ANN002, ANN003
        start = time.time()
        result = func(*args, **kwargs)
        stop = time.time()

        logger.info(f'Execution of {func.__name__} took {stop - start:.5}s.')

        return result

    return wrapper_measure_runtime


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
