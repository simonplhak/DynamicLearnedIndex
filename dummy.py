from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from bucket import DynamicBucket
from learned_index import LearnedIndex

if TYPE_CHECKING:
    from bucket import Bucket
    from config import IndexConfig


class DummyIndex(LearnedIndex):
    """A dummy index implementation that randomly stores the objects in the buckets."""

    def __init__(self, config: IndexConfig) -> None:
        super().__init__(config)

        """Number of buckets."""
        self.buckets = {i: DynamicBucket(config.bucket_shape, config.distance.metric) for i in range(config.n_buckets)}

        self.bucket_size = config.bucket_shape[0]

    def train(self, buckets: list[Bucket]) -> float:
        s = time.time()

        # Add the vectors to the new buckets
        for existing_bucket in buckets:
            bucket_data = existing_bucket.get_data()
            bucket_indexes = existing_bucket.get_ids()

            # Randomly assign vectors to buckets
            classes = torch.randint(self.config.n_buckets, (existing_bucket.get_n_objects(),))

            for i, new_child_bucket in self.buckets.items():
                new_child_bucket.insert_bulk(bucket_data[classes == i], bucket_indexes[np.where(classes == i)])

        return time.time() - s

    def insert(self, buckets: list[Bucket]) -> bool:
        total_n_objects = sum(b.get_n_objects() for b in buckets)

        # Predict to which bucket each vector belongs
        bucket_ids = torch.randint(self.config.n_buckets, (total_n_objects,))

        # Check that buckets do not overflow
        for i, n_objects_in_bucket in enumerate(torch.bincount(bucket_ids, minlength=self.config.n_buckets)):
            if n_objects_in_bucket + self.buckets[i].get_n_objects() > self.bucket_size:
                # Overflow detected, try to fix it
                offset = 0
                for j, bucket in self.buckets.items():
                    bucket_ids[offset : offset + bucket.get_free_space()] = j
                    offset += bucket.get_free_space()
                break

        # Check that buckets do not overflow
        for i, n_objects_in_bucket in enumerate(torch.bincount(bucket_ids, minlength=self.config.n_buckets)):
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

    def search(self, query: Tensor, k: int, nprobe: int) -> tuple[np.ndarray, np.ndarray, int]:
        _ = nprobe  # We do not use nprobe
        bucket_id = int(torch.randint(self.config.n_buckets, (1,)))
        return self.buckets[bucket_id].search(query, k, nprobe)
