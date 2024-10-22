from __future__ import annotations

from abc import abstractmethod
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
from faiss import merge_knn_results
from loguru import logger

from bucket import Bucket

if TYPE_CHECKING:
    from torch import Tensor

    from index import Index


class Framework:
    def __init__(
        self,
        index_class: type[Index],
        arity: int,
        bucket_shape: tuple[int, int],
        metric: int,
        keep_max: bool,
    ) -> None:
        # Fundamental properties
        self.buffer: Bucket = Bucket(bucket_shape, metric)
        self.levels: list[Index] = []

        # Tree properties
        self.arity = arity

        # Data properties
        self.dimensionality: int = bucket_shape[1]
        self.metric = metric
        self.keep_max = keep_max

        # Python/implementation-specific properties
        self.index_class: type[Index] = index_class
        self.bucket_shape: tuple[int, int] = bucket_shape

    def insert(self, X: Tensor, I: int) -> None:
        """Insert a single vector into the index."""
        assert X.shape == (self.dimensionality,)

        self.compact(X, I)

    @abstractmethod
    def compact(self, X: Tensor, I: int) -> None:
        raise NotImplementedError

    def _create_new_level(self, index: Index) -> None:
        self.levels.append(index)

    def _empty_upper_levels(self, current_level: int) -> None:
        # Empty all levels above the current level as the bucket objects are now in the new index and should be empty
        self.buffer.empty()
        for i in range(current_level - 1):
            self.levels[i].empty()

    def search(self, query: Tensor, k: int, nprobe: int) -> tuple[np.ndarray, np.ndarray]:
        """Search the index for k nearest neighbors.

        Search the index for the k nearest neighbors of a single query vector
        while probing nprobe buckets at each level.
        """
        assert query.shape == (self.dimensionality,)

        query = query.reshape((1, self.dimensionality))

        # Search the buffer if it is not empty
        searchable_levels = [self.buffer] if not self.buffer.is_empty() else []
        # Add all levels that exist
        searchable_levels += [l for l in self.levels if l.exists()]

        n_partial_results = len(searchable_levels)

        D_all, I_all = (
            np.zeros((n_partial_results, 1, k), dtype=np.float32),
            np.zeros((n_partial_results, 1, k), dtype=np.int64),
        )

        for i, level in enumerate(searchable_levels):
            D_all[i, :, :], I_all[i, :, :] = level.search(query, k, nprobe)

        return merge_knn_results(D_all, I_all, keep_max=self.keep_max)

    def get_n_objects(self) -> int:
        """Return the total number of objects in the index."""
        return self.buffer.get_n_objects() + sum(map(self.index_class.get_n_objects, self.levels))

    def get_buckets(self, till_layer: int) -> list[Bucket]:
        """Return a list of buckets from the top level to the specified level, including the buffer."""
        return [
            self.buffer,
            *list(chain.from_iterable(map(self.index_class.get_buckets, self.levels[:till_layer]))),
        ]

    def print_stats(self) -> None:
        bucket_size = self.bucket_shape[0]

        logger.info('Index stats:')
        logger.info(f' Buffer: {self.buffer.get_n_objects()} | {self.buffer.get_capacity()}')
        for i, level in enumerate(self.levels):
            hard_capacity_limit = bucket_size * (self.arity ** (i + 1))
            logger.info(
                f' Level {i}: {level.get_n_objects()} | {hard_capacity_limit} ({level.get_total_capacity_with_bucket_expansion()})',
            )
            for j, bucket in enumerate(level.get_buckets()):
                logger.info(f'  Bucket {j}: {bucket.get_n_objects()} | {bucket.get_capacity()}')
        logger.info(f' Total: {self.get_n_objects()} objects')
