from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from statistics import mean, median
from typing import TYPE_CHECKING

from loguru import logger
from torch import arange, empty, float32, int64, take_along_dim
from tqdm import tqdm

from dli import faiss_facade
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
            config.distance_function,
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
    def perform_search(
        self,
        db_size: int,
        config: SearchConfig,
        Q: Tensor,
        GT: Tensor,
    ) -> ExperimentSearchResult:
        # Bucket selection step
        s = time.time()

        n_candidates_per_query = empty(len(Q), dtype=float32)
        per_query_statistics: list[FrameworkSearchStatistics] = [None] * len(Q)  # type: ignore

        I = empty((len(Q), config.k), dtype=int64)
        recalls = empty(len(Q), dtype=float32)

        buckets_to_visit = None
        if config.search_strategy == ModelDrivenSearchStrategy:
            buckets_to_visit = self.select_buckets_model_driven(Q, config)
        if config.search_strategy == KNNSearchStrategy:
            buckets_to_visit = self.select_buckets_knn(Q, config)
        assert buckets_to_visit is not None

        bucket_selection_time = time.time() - s

        # Search step
        s = time.time()

        faiss_facade.set_num_threads(config.faiss_max_threads)

        with ThreadPoolExecutor(max_workers=config.python_max_workers) as executor:
            results = executor.map(
                lambda i: (
                    i,
                    self.search_single_query(
                        Q[i : i + 1],
                        buckets_to_visit[i],
                        config,
                    ),
                ),
                range(len(Q)),
            )

            iterator = tqdm(results, total=len(Q)) if config.verbose else results
            for i, (_, I_query, statistics) in iterator:
                I[i, :] = I_query
                n_candidates_per_query[i] = statistics.total_n_candidates
                per_query_statistics[i] = statistics

        search_time = time.time() - s

        # Compute recall
        for i in range(len(Q)):
            recalls[i] = len(set((I[i, :] + 1).tolist()).intersection(set(GT[i, : config.k].tolist()))) / config.k

        return ExperimentSearchResult(
            config,
            db_size,
            len(Q),
            recalls.sum().item(),
            n_candidates_per_query,
            per_query_statistics,
            bucket_selection_time,
            search_time,
            search_time + bucket_selection_time,
        )

    def select_buckets_knn(self, Q: Tensor, config: SearchConfig) -> list[list[Bucket]]:
        search_strategy = config.search_strategy(self.config.arity, 1 + len(self.levels))

        per_level_bucket_ids = []
        for level_idx, level in enumerate(self.levels):
            level_nprobe = search_strategy.determine_level_nprobe(level_idx, config.nprobe)
            bucket_ids = level.predict_top_k_bucket_probabilities(Q, level_nprobe)
            per_level_bucket_ids.append(bucket_ids)

        return [
            [
                self.levels[level_idx].buckets[bucket_idx]
                for level_idx, buckets_on_level in enumerate(per_level_bucket_ids)
                for bucket_idx in buckets_on_level[i, :].tolist()
            ]
            + [self.buffer]
            for i in range(len(Q))
        ]

    def select_buckets_model_driven(self, Q: Tensor, config: SearchConfig) -> list[list[Bucket]]:
        def _get_buckets_to_visit(i: int, level_ids: Tensor, bucket_ids: Tensor) -> list[Bucket]:
            return [
                self.levels[level_idx].buckets[bucket_idx]
                for level_idx, bucket_idx in zip(level_ids[i, :].tolist(), bucket_ids[i, :].tolist())
            ] + [self.buffer]

        # Collect bucket probabilities from each level
        n_buckets = self._get_n_buckets()

        bucket_probabilities = empty((len(Q), n_buckets), dtype=float32)
        level_index = empty((len(Q), n_buckets), dtype=int64)
        bucket_index = empty((len(Q), n_buckets), dtype=int64)
        offset = 0

        for level_idx, level in enumerate(self.levels):
            prediction = level.predict_bucket_probabilities(Q)
            bucket_probabilities[:, offset : offset + prediction.shape[1]] = prediction
            level_index[:, offset : offset + prediction.shape[1]] = level_idx
            bucket_index[:, offset : offset + prediction.shape[1]] = arange(prediction.shape[1])
            offset += prediction.shape[1]

        # Determine the order of buckets
        sorted_idxs = bucket_probabilities.argsort(dim=1, descending=True)
        level_ids = take_along_dim(level_index, sorted_idxs, dim=1)
        bucket_ids = take_along_dim(bucket_index, sorted_idxs, dim=1)

        level_ids = level_ids[:, : config.nprobe - 1]  # -1 because of the buffer
        bucket_ids = bucket_ids[:, : config.nprobe - 1]  # -1 because of the buffer

        return [_get_buckets_to_visit(i, level_ids, bucket_ids) for i in range(len(Q))]

    def insert(self, X: Tensor, I: int) -> FrameworkCompactionStatistics:
        """Insert a single vector into the index."""
        assert X.shape == (self.dimensionality,)

        return self.compact(X, I)

    def compact(self, X: Tensor, I: int) -> FrameworkCompactionStatistics:
        return self.compaction_strategy.compact(X, I)

    def search_single_query(
        self,
        query: Tensor,
        buckets_to_visit: list[Bucket],
        config: SearchConfig,
    ) -> tuple[Tensor, Tensor, FrameworkSearchStatistics]:
        """Search the buckets for k nearest neighbors.

        Parameters
        ----------
        query : Tensor
            Single query vector of shape (1, dimensionality).
        buckets_to_visit : list[Bucket]
            List of buckets to visit.
        config : SearchConfig
            Search configuration.

        Returns
        -------
        tuple[Tensor, Tensor, SearchStatistics]
            A tuple containing the neighbor distances, neighbor indices, and the search statistics.

        """
        s = time.time()

        # Prepare helper variables
        n_partial_results = len(buckets_to_visit)
        D_all, I_all = (
            empty((n_partial_results, 1, config.k), dtype=float32),
            empty((n_partial_results, 1, config.k), dtype=int64),
        )
        n_candidates = 0
        search_time_per_level_in_ms = [0.0] * n_partial_results
        preparation_time_in_ms = (time.time() - s) * SEC_TO_MSEC

        # Search the buckets
        s = time.time()
        for i, bucket in enumerate(buckets_to_visit):
            s = time.time()
            # TODO: -1
            D_all[i, :, :], I_all[i, :, :], n_level_candidates = bucket.search(query, config.k, -1)
            search_time_per_level_in_ms[i] += (time.time() - s) * SEC_TO_MSEC

            n_candidates += n_level_candidates
        search_time_in_ms = (time.time() - s) * SEC_TO_MSEC

        # Merge step
        s = time.time()
        D, I = merge_knn_results(D_all, I_all, keep_max=self.config.distance_function.keep_max_values)
        merge_time_in_ms = (time.time() - s) * SEC_TO_MSEC

        statistics = FrameworkSearchStatistics(
            total_n_candidates=n_candidates,
            search_time_per_level_in_ms=search_time_per_level_in_ms,
            preparation_time_in_ms=preparation_time_in_ms,
            search_time_in_ms=search_time_in_ms,
            merge_time_in_ms=merge_time_in_ms,
            total_search_time_in_ms=search_time_in_ms + preparation_time_in_ms + merge_time_in_ms,
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
