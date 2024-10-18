from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Literal

import numpy as np
from faiss import merge_knn_results

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

        self.compact(X, I, current_level=1)

    def compact(
        self,
        X: Tensor,
        I: int,
        current_level: int,
        # strategy: Literal['bentley_saxe'] | Literal['leveling'] = 'bentley_saxe',
        strategy: Literal['bentley_saxe'] | Literal['leveling'] = 'leveling',
    ) -> None:
        # TODO: use design pattern

        match strategy:
            case 'bentley_saxe':
                self.buffer.insert_single(X, I)

                # If the buffer is full, we need to merge it with the first level
                if self.buffer.is_full():
                    self.compact_bentley_saxe(current_level)
                return
            case 'leveling':
                if not self.buffer.is_full():
                    self.buffer.insert_single(X, I)
                    return

                self.compact_leveling(X, I)
                return

            # TODO: implement other compaction strategies
        assert False

    def compact_leveling(self, X: Tensor, I: int) -> None:
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

    def compact_bentley_saxe(self, current_level: int) -> None:
        # TODO: implement my own algorithm for situations where the clustering does not conform to maximal bucket sizes - essentially `train` method (is this framework or index specific?)
        """Compact the data from the buffer into the index.

        Compact the data from the buffer by either creating a new level
        or merging it with an existing one. The buffer will be emptied by this method.
        """
        # Take all data above this level and either create a new index or merge the data into an existing one

        # Option 1 -- We have descended beyond the existing levels, there is no index, we must create one
        if current_level > len(self.levels):
            index = self.index_class(
                n_buckets=pow(self.arity, current_level),
                metric=self.metric,
                bucket_shape=self.bucket_shape,
            )
            index.train(self.get_buckets(current_level - 1))
            self._create_new_level(index)
            self._empty_upper_levels(current_level)

        # Option 2 -- We are on a level that already exists
        else:
            current_index = self._get_index(current_level)

            # Option 2.1 -- The index does not exist at this level, we have to create it
            #            -- Actually, we are just training it because we have not thrown out the old one...
            if not current_index.exists():
                # TODO: reset the model's weights first
                current_index.train(self.get_buckets(current_level - 1))
                self._empty_upper_levels(current_level)

            # Option 2.2 -- The index exists
            else:
                is_successfully_inserted = current_index.insert(self.get_buckets(current_level - 1))

                # Option 2.2.1 -- All data has been stored at this level
                if is_successfully_inserted:
                    self._empty_upper_levels(current_level)
                    return

                # Option 2.2.2 -- This level is overflowing, try to fit the objects in the level below
                if not is_successfully_inserted:  # The level is not able to absorb the data = overflow detected
                    # Option 2.2.2.1 -- retrain(level) = if the total number of objects is less than BUCKET_SIZE -> we can retrain the model and reorganize the data
                    # TODO: implement
                    # Option 2.2.2.2 -- comact(level + 1) = if the total number of objects is equal to BUCKET_SIZE -> we have no other choice
                    self.compact_bentley_saxe(current_level + 1)
                    # Option 2.2.2.3 -- train for a bit using BLISS training procedure = use BLISS to reorganize the existing and new data
                    # TODO: implement

                    return

    def _create_new_level(self, index: Index) -> None:
        self.levels.append(index)

    def _empty_upper_levels(self, current_level: int) -> None:
        # Empty all levels above the current level as the bucket objects are now in the new index and should be empty
        self.buffer.empty()
        for i in range(current_level - 1):
            self.levels[i].empty()

    def search(self, query: Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search the index for the k nearest neighbors of a single query vector."""
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
            D_all[i, :, :], I_all[i, :, :] = level.search(query, k)

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

    def _get_index(self, level: int) -> Index:
        """Return the index at the specified level. Converts 1-based indexing to 0-based indexing."""
        return self.levels[level - 1]

    def print_stats(self) -> None:
        bucket_size = self.bucket_shape[0]

        print('Index stats:')
        print(f' Buffer: {self.buffer.get_n_objects()} | {bucket_size}')
        for i, level in enumerate(self.levels):
            print(f' Level {i}: {level.get_n_objects()} | {bucket_size * (self.arity ** (i+1))}')
            for j, bucket in enumerate(level.get_buckets()):
                print(f'  Bucket {j}: {bucket.get_n_objects()} | {bucket_size}')
        print(f' Total: {self.get_n_objects()} objects')
