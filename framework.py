from __future__ import annotations

import time
from abc import abstractmethod
from itertools import chain
from statistics import mean, median
from typing import TYPE_CHECKING

import numpy as np
from faiss import merge_knn_results
from loguru import logger

from bucket import Bucket
from framework_search_statistics import FrameworkSearchStatistics

if TYPE_CHECKING:
    from torch import Tensor

    from configuration import FrameworkConfig
    from framework_compaction_statistics import FrameworkCompactionStatistics
    from index import Index

SEC_TO_MSEC = 1_000


class Framework:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config

        # Fundamental properties
        self.buffer: Bucket = Bucket(config.bucket_shape, config.distance.metric)
        self.levels: list[Index] = []

        # Data properties
        self.dimensionality: int = config.bucket_shape[1]

    def insert(self, X: Tensor, I: int) -> FrameworkCompactionStatistics:
        """Insert a single vector into the index."""
        assert X.shape == (self.dimensionality,)

        return self.compact(X, I)

    @abstractmethod
    def compact(self, X: Tensor, I: int) -> FrameworkCompactionStatistics:
        raise NotImplementedError

    def _create_new_level(self, index: Index) -> None:
        self.levels.append(index)

    def _empty_upper_levels(self, current_level: int) -> None:
        # Empty all levels above the current level as the bucket objects are now in the new index and should be empty
        self.buffer.empty()
        for i in range(current_level - 1):
            self.levels[i].empty()

    def search(self, query: Tensor, k: int, nprobe: int) -> tuple[np.ndarray, np.ndarray, FrameworkSearchStatistics]:
        """Search the index for k nearest neighbors.

        Parameters
        ----------
        query : Tensor
            Single query vector of shape (dimensionality,).
        k : int
            Number of nearest neighbors to search for.
        nprobe : int
            Number of buckets to probe at each level.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, SearchStatistics]
            A tuple containing the neighbor distances, neighbor indices, and the search statistics.

        """
        assert query.shape == (self.dimensionality,)

        # Preparation step
        s = time.time()
        query = query.reshape((1, self.dimensionality))
        search_strategy = self.config.search_strategy(self.config.arity, 1 + len(self.levels))

        # Search the buffer if it is not empty
        searchable_levels = [self.buffer] if not self.buffer.is_empty() else []
        # Add all levels
        searchable_levels += self.levels

        n_partial_results = len(searchable_levels)

        D_all, I_all = (
            np.zeros((n_partial_results, 1, k), dtype=np.float32),
            np.zeros((n_partial_results, 1, k), dtype=np.int64),
        )
        n_candidates_per_level = [0] * n_partial_results
        search_time_per_level_in_ms = [0.0] * n_partial_results
        preparation_time_in_ms = (time.time() - s) * SEC_TO_MSEC

        # Search step
        s = time.time()
        for i, level in enumerate(searchable_levels):
            s = time.time()
            level_nprobe = search_strategy.determine_level_nprobe(i, nprobe)
            D_all[i, :, :], I_all[i, :, :], n_level_candidates = level.search(query, k, level_nprobe)
            search_time_per_level_in_ms[i] += (time.time() - s) * SEC_TO_MSEC

            n_candidates_per_level[i] += n_level_candidates
        search_time_in_ms = (time.time() - s) * SEC_TO_MSEC

        # Merge step
        s = time.time()
        D, I = merge_knn_results(D_all, I_all, keep_max=self.config.distance.keep_max)
        merge_time_in_ms = (time.time() - s) * SEC_TO_MSEC

        # Collect statistics

        ## Calculate result_object_level_location
        # ! Takes a long time, but is needed for debugging.
        # ! TODO: run only with debug flag
        # TODO: refactor + extract into a separate method
        # result_object_level_location: list[tuple[int, int]] = []  # (i, j) where i is the level and j is the bucket
        # nns = I[0]
        # assert len(nns) == k
        # for nn_id in nns.tolist():
        #     if nn_id in self.buffer.get_ids():
        #         result_object_level_location.append((-1, -1))  # -1, -1 ~ buffer
        #     else:
        #         for i, level in enumerate(self.levels):
        #             for j, bucket in enumerate(level.buckets.values()):
        #                 if nn_id in bucket.get_ids():
        #                     result_object_level_location.append((i, j))
        #                     break
        # assert len(result_object_level_location) == k
        result_object_level_location = []

        statistics = FrameworkSearchStatistics(
            total_n_candidates=sum(n_candidates_per_level),
            n_candidates_per_level=n_candidates_per_level,
            search_time_per_level_in_ms=search_time_per_level_in_ms,
            preparation_time_in_ms=preparation_time_in_ms,
            search_time_in_ms=search_time_in_ms,
            merge_time_in_ms=merge_time_in_ms,
            total_search_time_in_ms=search_time_in_ms + preparation_time_in_ms + merge_time_in_ms,
            result_object_level_location=result_object_level_location,
        )

        return D, I, statistics

    def get_n_objects(self) -> int:
        """Return the total number of objects in the index."""
        return self.buffer.get_n_objects() + sum(map(self.config.index_class.get_n_objects, self.levels))

    def get_buckets(self, till_layer: int) -> list[Bucket]:
        """Return a list of buckets from the top level to the specified level, including the buffer."""
        return [
            self.buffer,
            *list(chain.from_iterable(map(self.config.index_class.get_buckets, self.levels[:till_layer]))),
        ]

    def print_stats(self) -> None:
        bucket_size = self.config.bucket_shape[0]

        logger.info('Index stats:')
        logger.info(f' Buffer: {self.buffer.get_n_objects()} | {self.buffer.get_capacity()}')

        for i, level in enumerate(self.levels):
            hard_capacity_limit = bucket_size * (self.config.arity ** (i + 1))
            status = 'OK' if not level.is_degenerated() else 'DEGENERATED'

            logger.info(
                f' Level {i} ({status}): {level.get_n_objects()}'
                f' | {hard_capacity_limit} ({level.get_total_capacity_with_bucket_expansion()})',
            )

            for j, bucket in enumerate(level.get_buckets()):
                logger.info(f'  Bucket {j}: {bucket.get_n_objects()} | {bucket.get_capacity()}')

        logger.info(f' Total: {self.get_n_objects()} objects')

    def collect_stats(self) -> dict:
        return {
            'summary': {
                'n_objects': self.get_n_objects(),
                'n_levels': len(self.levels),
                'n_buckets': sum(map(len, map(self.config.index_class.get_buckets, self.levels))),
            },
            'buffer': {
                'n_objects': self.buffer.get_n_objects(),
                'capacity': self.buffer.get_capacity(),
            },
            'levels': [
                {
                    'n_objects': level.get_n_objects(),
                    'n_buckets': len(level.get_buckets()),
                    'total_capacity_with_bucket_expansion': level.get_total_capacity_with_bucket_expansion(),
                    'total_capacity': level.get_total_capacity(),
                    'is_degenerated': level.is_degenerated(),
                    'min_occupation': min(b.get_n_objects() for b in level.get_buckets()),
                    'avg_occupation': mean(b.get_n_objects() for b in level.get_buckets()) / len(level.get_buckets()),
                    'median_occupation': median(b.get_n_objects() for b in level.get_buckets()),
                    'max_occupation': max(b.get_n_objects() for b in level.get_buckets()),
                    'min_capacity': min(b.get_capacity() for b in level.get_buckets()),
                    'avg_capacity': mean(b.get_capacity() for b in level.get_buckets()) / len(level.get_buckets()),
                    'median_capacity': median(b.get_capacity() for b in level.get_buckets()),
                    'max_capacity': max(b.get_capacity() for b in level.get_buckets()),
                }
                for level in self.levels
            ],
        }
