from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from configuration import IndexConfig
from framework import Framework

if TYPE_CHECKING:
    from torch import Tensor

    from configuration import FrameworkConfig


class Leveling(Framework):
    def __init__(self, config: FrameworkConfig) -> None:
        super().__init__(config)

    def compact(
        self,
        X: Tensor,
        I: int,
    ) -> None:
        if not self.buffer.is_full():
            self.buffer.insert_single(X, I)
            return

        if len(self.levels) == 0:
            index = self.config.index_class(
                IndexConfig(
                    n_buckets=pow(self.config.arity, 1),
                    distance=self.config.distance,
                    bucket_shape=self.config.bucket_shape,
                    sample_percentage=self.config.sample_percentage,
                ),
            )
            index.train([self.buffer])
            self._create_new_level(index)
            self.buffer.empty()

            # Add the new vector into the buffer
            self.buffer.insert_single(X, I)
            return

        # Set to len(self.levels), so that we allocate a new level if no level with enough space is found
        idx = len(self.levels)  # idx in [0, len(self.levels))]

        for i in range(1, len(self.levels)):
            if self.levels[i].get_free_space() >= self.levels[i - 1].get_n_objects():
                idx = i  # Found a level with enough space
                break

        # idx == len(self.levels) -> Allocate a new level
        # idx != len(self.levels) -> We progressively merge existing levels and then accommodate the data

        for i in range(idx, 0, -1):
            if i == len(self.levels):  # We need to allocate a new level
                index = self.config.index_class(
                    IndexConfig(
                        n_buckets=pow(self.config.arity, i + 1),
                        distance=self.config.distance,
                        bucket_shape=self.config.bucket_shape,
                        sample_percentage=self.config.sample_percentage,
                    ),
                )
                index.train(self.levels[i - 1].get_buckets())
                self._create_new_level(index)
            else:
                # ? when to retrain and when to keep the existing model?
                logger.info(f'{self.levels[i].is_degenerated()=}')
                if self.levels[i].is_degenerated():  # Throw away the degenerated index and create a new one
                    # * Here, we could simply reset the model weights without allocating a new model
                    index = self.config.index_class(self.levels[i].config)

                    index.train([*self.levels[i].get_buckets(), *self.levels[i - 1].get_buckets()])
                    self.levels[i] = index
                else:  # Accommodate the data as the index is not degenerated
                    assert self.levels[i].insert(self.levels[i - 1].get_buckets())

            self.levels[i - 1].empty()

        # Accommodate the buffer data into the 0th level
        assert self.levels[0].insert([self.buffer])

        # Empty the buffer
        self.buffer.empty()

        # Add the new vector into the buffer
        self.buffer.insert_single(X, I)
