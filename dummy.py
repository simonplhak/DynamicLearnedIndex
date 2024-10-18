from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from index import Index

if TYPE_CHECKING:
    import numpy as np

    from bucket import Bucket


class DummyIndex(Index):
    """A dummy index implementation that randomly stores the objects in the buckets."""

    def __init__(
        self,
        n_buckets: int,
        metric: int,
        bucket_shape: tuple[int, int],
    ) -> None:
        super().__init__(
            n_buckets,
            metric,
            bucket_shape,
        )

        self.bucket_size = bucket_shape[0]

    def train(self, buckets: list[Bucket]) -> None:
        # Add the vectors to the new buckets
        for existing_bucket in buckets:
            bucket_data = existing_bucket.get_data()
            bucket_indexes = existing_bucket.get_ids()

            # Randomly assign vectors to buckets
            classes = torch.randint(self.n_buckets, (existing_bucket.get_n_objects(),))

            for i, new_child_bucket in self.buckets.items():
                new_child_bucket.insert_bulk(bucket_data[classes == i], bucket_indexes[classes == i])

        self.is_trained = True

    def insert(self, buckets: list[Bucket]) -> bool:
        total_n_objects = sum(b.get_n_objects() for b in buckets)

        # Predict to which bucket each vector belongs
        bucket_ids = torch.randint(self.n_buckets, (total_n_objects,))

        # Check that buckets do not overflow
        for i, n_objects_in_bucket in enumerate(torch.bincount(bucket_ids, minlength=self.n_buckets)):
            if n_objects_in_bucket + self.buckets[i].get_n_objects() > self.bucket_size:
                # Overflow detected, try to fix it
                offset = 0
                for j, bucket in self.buckets.items():
                    bucket_ids[offset : offset + bucket.get_free_space()] = j
                    offset += bucket.get_free_space()
                break

        # Check that buckets do not overflow
        for i, n_objects_in_bucket in enumerate(torch.bincount(bucket_ids, minlength=self.n_buckets)):
            if n_objects_in_bucket + self.buckets[i].get_n_objects() > self.bucket_size:
                return False  # Overflow detected

        self._assign_objects_to_new_buckets(bucket_ids, buckets)

        return True  # Insertion successful

    def _assign_objects_to_new_buckets(self, bucket_ids: Tensor, buckets: list[Bucket]) -> None:
        # Add the vectors to the new buckets
        offset = 0

        for existing_bucket in buckets:
            bucket_data = existing_bucket.get_data()
            bucket_indexes = existing_bucket.get_ids()
            relevant_bucket_ids = bucket_ids[offset : offset + existing_bucket.get_n_objects()]

            for i, new_child_bucket in self.buckets.items():
                new_child_bucket.insert_bulk(
                    bucket_data[relevant_bucket_ids == i],
                    bucket_indexes[relevant_bucket_ids == i],
                )

            offset += existing_bucket.get_n_objects()

    def search(self, query: Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
        bucket_id = int(torch.randint(self.n_buckets, (1,)))
        return self.buckets[bucket_id].search(query, k)
