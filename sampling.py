from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from bucket import Bucket

SEED = 42

# Set seeds for reproducibility
np_rng = np.random.default_rng(SEED)
torch_rng = torch.Generator().manual_seed(SEED)


def take_sample(buckets: list[Bucket], sample_size: int, dimensionality: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a tensor of size (sample_size, dimensionality) with the sample of objects from the given buckets."""
    total_n_objects = sum(b.get_n_objects() for b in buckets)
    sample_indexes = torch.randint(total_n_objects, (sample_size,))

    start, stop, result_offset = 0, 0, 0
    result = torch.empty((sample_size, dimensionality), dtype=torch.float32)

    for bucket in buckets:
        stop += bucket.get_n_objects()

        bucket_sample_indexes = sample_indexes[(start <= sample_indexes) & (sample_indexes < stop)] - start

        result[result_offset : result_offset + len(bucket_sample_indexes)] = bucket.get_data()[bucket_sample_indexes]

        result_offset += len(bucket_sample_indexes)
        start += bucket.get_n_objects()

    return result, sample_indexes
