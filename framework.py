from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

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
        self.top_level_bucket: Bucket = Bucket(bucket_shape, metric)  # TODO: rename to (mutable?) buffer
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

        # Add the vector to the top level bucket
        self.top_level_bucket.insert_single(X, I)

        # If the top level bucket is full, we need to merge it with the first level
        if self.top_level_bucket.is_full():
            self.compact(current_level=1)

    def compact(self, current_level: int) -> None:  # TODO: implement other compaction strategies
        """Compact the data from the top level into the index.

        Compact the data from the top level bucket by either creating a new level
        or merging it with an existing one. The top level bucket will be emptied by this method.
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
                    self.compact(current_level + 1)
                    # Option 2.2.2.3 -- train for a bit using BLISS training procedure = use BLISS to reorganize the existing and new data
                    # TODO: implement

                    return

    def _create_new_level(self, index: Index) -> None:
        self.levels.append(index)

    def _empty_upper_levels(self, current_level: int) -> None:
        # Empty all levels above the current level as the bucket objects are now in the new index and should be empty
        self.top_level_bucket.empty()
        for i in range(current_level - 1):
            self.levels[i].empty()

    def search(self, query: Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search the index for the k nearest neighbors of a single query vector."""
        assert query.shape == (self.dimensionality,)

        query = query.reshape((1, self.dimensionality))

        # Search the top level if it is not empty
        searchable_levels = [self.top_level_bucket] if not self.top_level_bucket.is_empty() else []
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
        return self.top_level_bucket.get_n_objects() + sum(map(self.index_class.get_n_objects, self.levels))

    def get_buckets(self, till_layer: int) -> list[Bucket]:
        """Return a list of buckets from the top level to the specified level."""
        return [
            self.top_level_bucket,
            *list(chain.from_iterable(map(self.index_class.get_buckets, self.levels[:till_layer]))),
        ]

    def _get_index(self, level: int) -> Index:
        """Return the index at the specified level. Converts 1-based indexing to 0-based indexing."""
        return self.levels[level - 1]

    def print_stats(self) -> None:
        bucket_size = self.bucket_shape[0]

        print('Index stats:')
        print(f' Top level: {self.top_level_bucket.get_n_objects()} | {bucket_size}')
        for i, level in enumerate(self.levels):
            print(f' Level {i + 1}: {level.get_n_objects()} | {bucket_size * (self.arity ** (i+1))}')
            for j, bucket in enumerate(level.get_buckets()):
                print(f'  Bucket {j}: {bucket.get_n_objects()} | {bucket_size}')
        print(f' Total: {self.get_n_objects()} objects')
