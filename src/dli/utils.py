from __future__ import annotations

import datetime
import functools
import subprocess
import time
from typing import TYPE_CHECKING, Callable, ParamSpec, TypeVar

import psutil
from loguru import logger

if TYPE_CHECKING:
    from torch.nn import Sequential


Param = ParamSpec('Param')
ReturnType = TypeVar('ReturnType')


def sizeof_fmt(num: float, suffix: str = 'B') -> str:
    """Format a size in bytes to a human readable string.

    Adapted from:
    https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size

    Args:
        num: Size in bytes
        suffix: Unit suffix to append

    Returns:
        Formatted string like '3.1GiB'

    """
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:  # noqa: PLR2004
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


def time_fmt(seconds: float) -> str:
    """Format a time in seconds to a human readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like '1 day, 18:20:07.732870'

    """
    return str(datetime.timedelta(seconds=seconds))


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


def measure_memory_usage(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    """Measure and log the memory usage of a function.

    Args:
        func: Function to measure

    Returns:
        Wrapped function that logs memory usage

    """

    @functools.wraps(func)
    def wrapper_measure_memory_usage(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
        process = psutil.Process()
        start = process.memory_info().rss
        result = func(*args, **kwargs)
        stop = process.memory_info().rss
        logger.debug(f'Function {func.__name__} allocated {sizeof_fmt(stop - start)}.')
        return result

    return wrapper_measure_memory_usage


def obtain_commit_hash() -> str:
    """Get the current git commit hash."""
    try:
        return subprocess.check_output(['git', 'describe', '--always'], text=True).strip()  # noqa: S603, S607
    except Exception:  # noqa: BLE001
        return 'unknown'


def obtain_dirty_state() -> bool:
    """Check if the git repository has uncommitted changes."""
    try:
        return subprocess.call(['git', 'diff', '--quiet']) != 0  # noqa: S603, S607
    except Exception:  # noqa: BLE001
        return True


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
