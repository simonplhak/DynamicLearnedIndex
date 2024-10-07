from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from index import Index

if TYPE_CHECKING:
    from bucket import Bucket


class DummyIndex(Index):
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
        # TODO: get rid of torch.concatenate
        X, I = torch.concatenate([b.get_data() for b in buckets]), np.concatenate([b.get_ids() for b in buckets])

        # Randomly assign vectors to buckets
        classes = torch.randint(self.n_buckets, (X.shape[0],))

        # Add the vectors to the buckets
        for i, child_bucket in self.buckets.items():
            child_bucket.insert(X[classes == i], I[classes == i])

        self.is_trained = True

        # TODO: what if all objects were inserted into the same bucket? = unbalanced partitioning

    def insert(self, buckets: list[Bucket]) -> bool:
        # TODO: get rid of torch.concatenate
        X, I = torch.concatenate([b.get_data() for b in buckets]), np.concatenate([b.get_ids() for b in buckets])

        # Predict to which bucket each vector belongs
        bucket_ids = torch.randint(self.n_buckets, (X.shape[0],))

        # Check that buckets do not overflow
        for i, n_objects_in_bucket in enumerate(torch.bincount(bucket_ids, minlength=self.n_buckets)):
            if n_objects_in_bucket + self.buckets[i].get_n_objects() > self.bucket_size:
                return False  # Overflow detected

        # Add the vectors to the buckets
        for i, child_bucket in self.buckets.items():
            child_bucket.insert(X[bucket_ids == i], I[bucket_ids == i])

        return True  # Insertion successful

    def search(self, query: Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
        bucket_id = int(torch.randint(self.n_buckets, (1,)))
        return self.buckets[bucket_id].search(query, k)
