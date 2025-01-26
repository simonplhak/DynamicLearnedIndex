from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Generator, Tensor, empty, float32, randperm

if TYPE_CHECKING:
    from dli.bucket import Bucket

SEED = 42

# Set seeds for reproducibility
torch_rng = Generator().manual_seed(SEED)


def take_sample(buckets: list[Bucket], sample_threshold: int, dimensionality: int) -> tuple[Tensor, Tensor]:
    """Return a tensor of size (sample_size, dimensionality) with the sample of objects from the given buckets."""
    total_n_objects = sum(b.get_n_objects() for b in buckets)
    sample_size = min(sample_threshold, total_n_objects)

    sample_indexes = randperm(total_n_objects, generator=torch_rng)[:sample_size]

    start, stop, result_offset = 0, 0, 0
    result = empty((sample_size, dimensionality), dtype=float32)

    for bucket in buckets:
        stop += bucket.get_n_objects()

        bucket_sample_indexes = sample_indexes[(start <= sample_indexes) & (sample_indexes < stop)] - start

        result[result_offset : result_offset + len(bucket_sample_indexes)] = bucket.get_data()[bucket_sample_indexes]

        result_offset += len(bucket_sample_indexes)
        start += bucket.get_n_objects()

    return result, sample_indexes
