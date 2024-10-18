from __future__ import annotations

from typing import TYPE_CHECKING

from framework import Framework

if TYPE_CHECKING:
    from torch import Tensor

    from index import Index


class Leveling(Framework):
    def __init__(
        self,
        index_class: type[Index],
        arity: int,
        bucket_shape: tuple[int, int],
        metric: int,
        keep_max: bool,
    ) -> None:
        super().__init__(index_class, arity, bucket_shape, metric, keep_max)

    def compact(
        self,
        X: Tensor,
        I: int,
    ) -> None:
        if not self.buffer.is_full():
            self.buffer.insert_single(X, I)
            return

        if len(self.levels) == 0:
            index = self.index_class(
                n_buckets=pow(self.arity, 1),
                metric=self.metric,
                bucket_shape=self.bucket_shape,
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
        # idx != len(self.levels) -> We do the merging of the existing levels and then accommodate the data

        for i in range(idx, 0, -1):
            if i == len(self.levels):  # We need to allocate a new level
                # TODO: reset the model's weights first?
                index = self.index_class(
                    n_buckets=pow(self.arity, i + 1),
                    metric=self.metric,
                    bucket_shape=self.bucket_shape,
                )
                index.train(self.levels[i - 1].get_buckets())
                self._create_new_level(index)
            else:
                assert self.levels[i].insert(self.levels[i - 1].get_buckets())

            self.levels[i - 1].empty()

        # Accommodate the buffer data into the 0th level
        assert self.levels[0].insert([self.buffer])

        # Empty the buffer
        self.buffer.empty()

        # Add the new vector into the buffer
        self.buffer.insert_single(X, I)
