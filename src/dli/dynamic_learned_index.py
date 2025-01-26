from __future__ import annotations

import time
from itertools import chain
from statistics import mean, median
from typing import TYPE_CHECKING

from loguru import logger
from torch import empty, float32, int64
from tqdm import tqdm

from dli.bucket import StaticBucket
from dli.faiss_facade import merge_knn_results
from dli.result import BuildResult, ExperimentSearchResult
from dli.search_strategy import KNNSearchStrategy, ModelDrivenSearchStrategy
from dli.statistic import FrameworkCompactionStatistics, FrameworkSearchStatistics
from dli.utils import measure_memory_usage, measure_runtime

if TYPE_CHECKING:
    from torch import Tensor

    from dli.bucket import Bucket
    from dli.config import DLIConfig, SearchConfig
    from dli.learned_index import LearnedIndex

SEC_TO_MSEC = 1_000


class DynamicLearnedIndex:
    def __init__(self, config: DLIConfig) -> None:
        self.config = config

        # Fundamental properties
        self.buffer = StaticBucket(
            config.bucket_shape,
            config.distance.distance_function,
            config.shrink_buckets_during_compaction,
        )
        self.levels: list[LearnedIndex] = []
        self.compaction_strategy = config.compaction_strategy(config, self)

        # Data properties
        self.dimensionality: int = config.bucket_shape[1]

    @measure_runtime
    @measure_memory_usage
    def insert_objects_sequentially(self, X: Tensor) -> BuildResult:
        """Insert the dataset one object at a time."""
        s = time.time()
        per_objects_insertion_statistics = []
        for i in range(len(X)):
            statistics = self.insert(X[i], i)
            per_objects_insertion_statistics.append(statistics)

            if (i + 1) % (len(X) // 10) == 0:
                logger.info(f'Inserted {((i + 1) / len(X) * 100):.0f}% ({i + 1}) objects')
                logger.info(f'Allocated memory: {self.measure_total_allocated_memory() / 1024**2:.0f} MB')

            assert self.get_n_objects() == i + 1, f'Wrong number of objects: {self.get_n_objects()} != {i + 1}'
        build_time = time.time() - s

        logger.info(f'Inserted {len(X)} objects')

        return BuildResult(build_time, self.collect_stats(), per_objects_insertion_statistics)

    @measure_runtime
    def perform_search(self, db_size: int, config: SearchConfig, Q: Tensor, GT: Tensor) -> ExperimentSearchResult:
        sum_of_recalls = 0.0
        n_candidates_per_query = empty(len(Q), dtype=float32)
        per_query_statistics: list[FrameworkSearchStatistics] = [None] * len(Q)  # type: ignore

        s = time.time()
        for i in tqdm(range(len(Q))):
            _, I, statistics = self.search_single(Q[i], config)

            recall = len(set((I[0] + 1).tolist()).intersection(set(GT[i, : config.k].tolist()))) / config.k

            sum_of_recalls += recall
            n_candidates_per_query[i] = statistics.total_n_candidates
            per_query_statistics[i] = statistics
        search_time = time.time() - s

        return ExperimentSearchResult(
            config,
            db_size,
            len(Q),
            sum_of_recalls,
            n_candidates_per_query,
            search_time,
            per_query_statistics,
        )

    def insert(self, X: Tensor, I: int) -> FrameworkCompactionStatistics:
        """Insert a single vector into the index."""
        assert X.shape == (self.dimensionality,)

        return self.compact(X, I)

    def compact(self, X: Tensor, I: int) -> FrameworkCompactionStatistics:
        return self.compaction_strategy.compact(X, I)

    def search_single(
        self,
        query: Tensor,
        config: SearchConfig,
    ) -> tuple[Tensor, Tensor, FrameworkSearchStatistics]:
        """Search the index for k nearest neighbors.

        Parameters
        ----------
        query : Tensor
            Single query vector of shape (dimensionality,).
        config : SearchConfig
            Search configuration.

        Returns
        -------
        tuple[Tensor, Tensor, SearchStatistics]
            A tuple containing the neighbor distances, neighbor indices, and the search statistics.

        """
        assert query.shape == (self.dimensionality,)

        if config.search_strategy == KNNSearchStrategy:
            return self.search_knn(query, config)

        if config.search_strategy == ModelDrivenSearchStrategy:
            return self.search_model_driven(query, config)

        raise NotImplementedError

    def search_knn(
        self,
        query: Tensor,
        config: SearchConfig,
    ) -> tuple[Tensor, Tensor, FrameworkSearchStatistics]:
        """Search the index for k nearest neighbors.

        Parameters
        ----------
        query : Tensor
            Single query vector of shape (dimensionality,).
        config : SearchConfig
            Search configuration.

        Returns
        -------
        tuple[Tensor, Tensor, SearchStatistics]
            A tuple containing the neighbor distances, neighbor indices, and the search statistics.

        """
        assert config.search_strategy == KNNSearchStrategy

        # Preparation step
        s = time.time()
        query = query.reshape((1, self.dimensionality))
        search_strategy = config.search_strategy(self.config.arity, 1 + len(self.levels))

        # Search the buffer if it is not empty
        searchable_levels = [self.buffer] if not self.buffer.is_empty() else []
        # Add all levels
        searchable_levels += self.levels

        n_partial_results = len(searchable_levels)

        D_all, I_all = (
            empty((n_partial_results, 1, config.k), dtype=float32),
            empty((n_partial_results, 1, config.k), dtype=int64),
        )
        n_candidates_per_level = [0] * n_partial_results
        search_time_per_level_in_ms = [0.0] * n_partial_results
        preparation_time_in_ms = (time.time() - s) * SEC_TO_MSEC

        # Search step
        s = time.time()
        for i, level in enumerate(searchable_levels):
            s = time.time()
            level_nprobe = search_strategy.determine_level_nprobe(i, config.nprobe)
            D_all[i, :, :], I_all[i, :, :], n_level_candidates = level.search(query, config.k, level_nprobe)
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

    def search_model_driven(
        self,
        query: Tensor,
        config: SearchConfig,
    ) -> tuple[Tensor, Tensor, FrameworkSearchStatistics]:
        """Search the index for k nearest neighbors.

        Parameters
        ----------
        query : Tensor
            Single query vector of shape (dimensionality,).
        config : SearchConfig
            Search configuration.

        Returns
        -------
        tuple[Tensor, Tensor, SearchStatistics]
            A tuple containing the neighbor distances, neighbor indices, and the search statistics.

        """
        assert config.search_strategy == ModelDrivenSearchStrategy

        # Preparation step
        s = time.time()
        query = query.reshape((1, self.dimensionality))

        # Collect bucket scores from each level
        per_level_bucket_scores = [
            (level_idx, *bucket_scores)
            for level_idx, level in enumerate(self.levels)
            for bucket_scores in level.predict_bucket_scores(query)
        ]

        # Determine the order of buckets
        visit_order = sorted(per_level_bucket_scores, key=lambda x: x[2], reverse=True)

        # Collect the buckets
        bucket_locations_to_visit = visit_order[: config.nprobe - 1]
        buckets_to_visit = (
            [self.levels[level_idx].buckets[bucket_idx] for level_idx, bucket_idx, _ in bucket_locations_to_visit]
            + [self.buffer]
            if not self.buffer.is_empty()
            else []
        )

        # Prepare helper variables
        n_partial_results = len(buckets_to_visit)
        D_all, I_all = (
            empty((n_partial_results, 1, config.k), dtype=float32),
            empty((n_partial_results, 1, config.k), dtype=int64),
        )
        n_candidates_per_level = [0] * n_partial_results
        search_time_per_level_in_ms = [0.0] * n_partial_results
        preparation_time_in_ms = (time.time() - s) * SEC_TO_MSEC

        # Search the buckets
        s = time.time()
        for i, bucket in enumerate(buckets_to_visit):
            s = time.time()
            D_all[i, :, :], I_all[i, :, :], n_level_candidates = bucket.search(query, config.k, -1)
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
                'n_buckets': self._get_n_buckets(),
                'n_empty_buckets': self._get_n_empty_buckets(),
                'total_allocated_memory_in_mb': int(self.measure_total_allocated_memory() / 1024**2),
                'total_allocated_memory_for_models_in_mb': int(self.measure_allocated_model_memory() / 1024**2),
                'total_allocated_memory_for_buckets_in_mb': int(self.measure_allocated_bucket_memory() / 1024**2),
                'overall_bucket_space_utilization': self.calculate_bucket_space_utilization(),
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
                    'is_degenerated': level.is_degenerated(),  # ? replace?
                    'n_empty_buckets': level.get_n_empty_buckets(),
                    'percentage_of_empty_buckets': level.get_percentage_of_empty_buckets(),
                    'min_occupation': min(b.get_n_objects() for b in level.get_buckets()),
                    'avg_occupation': mean(b.get_n_objects() for b in level.get_buckets()),
                    'median_occupation': median(b.get_n_objects() for b in level.get_buckets()),
                    'max_occupation': max(b.get_n_objects() for b in level.get_buckets()),
                    'min_capacity': min(b.get_capacity() for b in level.get_buckets()),
                    'avg_capacity': mean(b.get_capacity() for b in level.get_buckets()),
                    'median_capacity': median(b.get_capacity() for b in level.get_buckets()),
                    'max_capacity': max(b.get_capacity() for b in level.get_buckets()),
                    'bucket_space_utilization': level.calculate_bucket_space_utilization(),
                }
                for level in self.levels
            ],
        }

    def measure_total_allocated_memory(self) -> int:
        """Return the total allocated memory of models and buckets in bytes."""
        return self.measure_allocated_model_memory() + self.measure_allocated_bucket_memory()

    def measure_allocated_model_memory(self) -> int:
        """Return the total allocated memory of models in bytes."""
        return sum([index.measure_allocated_model_memory() for index in self.levels])

    def measure_allocated_bucket_memory(self) -> int:
        """Return the total allocated memory of the buckets in bytes."""
        return self.buffer.get_allocated_memory() + sum(
            [index.measure_allocated_bucket_memory() for index in self.levels],
        )

    def calculate_bucket_space_utilization(self) -> float:
        """Return the utilization of the index."""
        total_bucket_capacity = self.buffer.get_capacity() + sum(
            map(self.config.index_class.get_total_capacity_with_bucket_expansion, self.levels),
        )
        return self.get_n_objects() / total_bucket_capacity

    def _get_n_buckets(self) -> int:
        """Return the total number of buckets in the index without the buffer."""
        return sum(map(self.config.index_class.get_n_buckets, self.levels))

    def _get_n_empty_buckets(self) -> int:
        """Return the total number of empty buckets in the index without the buffer."""
        return sum(map(self.config.index_class.get_n_empty_buckets, self.levels))
