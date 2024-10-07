import functools
import time
from typing import Any, Callable

import numpy as np
import torch

SEED = 42

# Set seeds for reproducibility
np_rng = np.random.default_rng(SEED)
torch_rng = torch.Generator().manual_seed(SEED)


def measure_runtime(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper_measure_runtime(*args, **kwargs) -> Any:  # noqa: ANN401, ANN002, ANN003
        start = time.time()
        result = func(*args, **kwargs)
        stop = time.time()

        print(f'Execution of {func.__name__} took {stop - start:.5}s.')

        return result

    return wrapper_measure_runtime
