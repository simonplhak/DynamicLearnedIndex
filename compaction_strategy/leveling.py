from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger

from config.index import IndexConfig
from statistic import FrameworkCompactionStatistics

if TYPE_CHECKING:
    from torch import Tensor

    from config.dli import DLIConfig
    from dynamic_learned_index import DynamicLearnedIndex

INSERTION_FAILED_MSG = 'Leveling: Insertion failed'


class Leveling:
    def __init__(self, config: DLIConfig, dli: DynamicLearnedIndex) -> None:
        self.dli = dli
        self.config = config

    def compact(self, X: Tensor, I: int) -> FrameworkCompactionStatistics:
        s = time.time()

        if not self.dli.buffer.is_full():
            self.dli.buffer.insert_single(X, I)
            return FrameworkCompactionStatistics(
                total_model_training_time=0.0,
                total_compaction_time=time.time() - s,
                allocated_new_level=False,
                n_retrained_indexes=0,
            )

        if len(self.dli.levels) == 0:
            index = self.dli.config.index_class(
                IndexConfig(
                    n_buckets=pow(self.dli.config.arity, 1),
                    distance=self.dli.config.distance,
                    bucket_shape=self.dli.config.bucket_shape,
                    sample_threshold=self.dli.config.sample_threshold,
                    n_training_samples=self.dli.buffer.get_n_objects(),
                ),
            )
            total_model_training_time = index.train([self.dli.buffer])
            self.dli.levels.append(index)
            self.dli.buffer.empty()

            # Add the new vector into the buffer
            self.dli.buffer.insert_single(X, I)
            return FrameworkCompactionStatistics(
                total_model_training_time,
                total_compaction_time=time.time() - s,
                allocated_new_level=True,
                n_retrained_indexes=0,
            )

        # Set to len(self.dli.levels), so that we allocate a new level if no level with enough space is found
        idx = len(self.dli.levels)  # idx in [0, len(self.dli.levels))]

        for i in range(1, len(self.dli.levels)):
            if self.dli.levels[i].get_free_space() >= self.dli.levels[i - 1].get_n_objects():
                idx = i  # Found a level with enough space
                break

        # idx == len(self.dli.levels) -> Allocate a new level
        # idx != len(self.dli.levels) -> We progressively merge existing levels and then accommodate the data

        total_model_training_time = 0.0
        n_retrained_indexes = 0
        allocated_new_level = False

        for i in range(idx, 0, -1):
            if i == len(self.dli.levels):  # We need to allocate a new level
                logger.info(f'Allocating new level {i}')
                index = self.dli.config.index_class(
                    IndexConfig(
                        n_buckets=pow(self.dli.config.arity, i + 1),
                        distance=self.dli.config.distance,
                        bucket_shape=self.dli.config.bucket_shape,
                        sample_threshold=self.dli.config.sample_threshold,
                        n_training_samples=self.dli.levels[i - 1].get_n_objects(),
                    ),
                )
                total_model_training_time += index.train(self.dli.levels[i - 1].get_buckets())
                self.dli.levels.append(index)

                allocated_new_level = True
            # ? when to retrain and when to keep the existing model?
            elif self.dli.levels[i].is_degenerated():  # Throw away the degenerated index and create a new one
                # * Here, we could simply reset the model weights without allocating a new model
                index = self.dli.config.index_class(self.dli.levels[i].config)

                total_model_training_time += index.train(
                    [*self.dli.levels[i].get_buckets(), *self.dli.levels[i - 1].get_buckets()],
                )
                self.dli.levels[i] = index

                n_retrained_indexes += 1
            else:  # Accommodate the data as the index is not degenerated
                inserted = self.dli.levels[i].insert(self.dli.levels[i - 1].get_buckets())
                if not inserted:
                    raise ValueError(INSERTION_FAILED_MSG)

            self.dli.levels[i - 1].empty()

        # Accommodate the buffer data into the 0th level
        inserted = self.dli.levels[0].insert([self.dli.buffer])
        if not inserted:
            raise ValueError(INSERTION_FAILED_MSG)

        # Empty the buffer
        self.dli.buffer.empty()

        # Add the new vector into the buffer
        self.dli.buffer.insert_single(X, I)

        return FrameworkCompactionStatistics(
            total_model_training_time,
            time.time() - s,
            allocated_new_level,
            n_retrained_indexes,
        )
