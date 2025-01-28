from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import h5py
from loguru import logger
from torch import Tensor, float16, float32, from_numpy, int32


@dataclass
class DatasetConfig:
    identifier: str
    """Identifier of the dataset."""
    size: int
    """Number of objects in the dataset."""
    X: Path
    """Path to the dataset."""
    Q: Path
    """Path to the queries."""
    GT: Path
    """Path to the ground truth."""


SUPPORTED_DATASETS = {
    '300K': DatasetConfig(
        '300K',
        300_000,
        Path('laion2B-en-clip768v2-n=300K.h5'),
        Path('public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
        Path('gold-standard-dbsize=300k--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
    ),
    '10M': DatasetConfig(
        '10M',
        10_120_191,
        Path('laion2B-en-clip768v2-n=10M.h5'),
        Path('public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
        Path('gold-standard-dbsize=10M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
    ),
    '100M': DatasetConfig(
        '100M',
        102_144_212,
        Path('laion2B-en-clip768v2-n=100M.h5'),
        Path('public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
        Path('gold-standard-dbsize=100M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
    ),
}

Param = ParamSpec('Param')
ReturnType = TypeVar('ReturnType')


def measure_runtime(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    """Measure and log the runtime of a function.

    Args:
        func: Function to measure

    Returns:
        Wrapped function that logs runtime

    """

    @functools.wraps(func)
    def wrapper_measure_runtime(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
        start = time.perf_counter()  # More precise than time.time()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start

        logger.debug(f'Execution of {func.__name__} took {duration:.5f}s.')

        return result

    return wrapper_measure_runtime


@measure_runtime
def load_data(config: DatasetConfig, path_prefix: Path) -> tuple[Tensor, Tensor, Tensor]:
    """Load dataset tensors from HDF5 files.

    Args:
        config: Dataset configuration containing file paths
        path_prefix: Path prefix for the dataset on the given execution environment
    Returns:
        Tuple of (X, Q, GT) tensors

    """

    def load_h5_tensor(path: Path, dataset: str, size: int | None = None) -> Tensor:
        with h5py.File(path_prefix / path, 'r') as f:
            data = f[dataset][:size] if size else f[dataset][:]  # type: ignore
            return from_numpy(data)

    X = load_h5_tensor(config.X, 'emb', config.size)
    Q = load_h5_tensor(config.Q, 'emb')
    GT = load_h5_tensor(config.GT, 'knns')

    assert X.dtype == float16, f'Expected X dtype float16, got {X.dtype}'
    assert Q.dtype == float32, f'Expected Q dtype float32, got {Q.dtype}'
    assert GT.dtype == int32, f'Expected GT dtype int32, got {GT.dtype}'

    return X, Q, GT
