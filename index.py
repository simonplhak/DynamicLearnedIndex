from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from bucket import Bucket

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from configuration import IndexConfig


class Index(ABC):
    def __init__(self, config: IndexConfig) -> None:
        self.config = config

        self.buckets: dict[int, Bucket]

    @abstractmethod
    def train(self, buckets: list[Bucket]) -> None:
        """Train the index on the objects in the given buckets."""
        raise NotImplementedError

    @abstractmethod
    def insert(self, buckets: list[Bucket]) -> bool:
        """Insert the objects from the given buckets into the index.

        Returns True if the insertion was successful.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query: Tensor, k: int, nprobe: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for the k nearest neighbors of the given query in the index."""
        raise NotImplementedError

    def empty(self) -> None:
        for bucket in self.buckets.values():
            bucket.empty()

    def get_n_objects(self) -> int:
        return sum(map(Bucket.get_n_objects, self.buckets.values()))

    def get_total_capacity_with_bucket_expansion(self) -> int:
        return sum(map(Bucket.get_capacity, self.buckets.values()))

    def get_total_capacity(self) -> int:
        """Return the expected total capacity of the index. Does NOT include bucket enlargements."""
        return self.config.n_buckets * self.config.bucket_shape[0]

    def get_free_space(self) -> int:
        return self.get_total_capacity() - self.get_n_objects()

    def get_buckets(self) -> list[Bucket]:
        return list(self.buckets.values())
