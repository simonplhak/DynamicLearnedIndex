from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from bucket import Bucket

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor


class Index(ABC):
    def __init__(
        self,
        n_buckets: int,
        bucket_shape: tuple[int, int],
    ) -> None:
        self.bucket_size = bucket_shape[0]
        """Size of each bucket."""
        self.dimensionality: int = bucket_shape[1]
        """Dimensionality of the data."""
        self.n_buckets: int = n_buckets
        """Number of buckets."""
        self.buckets: dict[int, Bucket]

        self.is_trained: bool = False

    @abstractmethod
    def train(self, buckets: list[Bucket]) -> None:
        """Train the index on the objects in the given buckets.

        The method must set self.is_trained to True.
        """
        raise NotImplementedError

    @abstractmethod
    def insert(self, buckets: list[Bucket]) -> bool:
        """Insert the objects from the given buckets into the index.

        Returns True if the insertion was successful.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query: Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for the k nearest neighbors of the given query in the index."""
        raise NotImplementedError

    def exists(self) -> bool:
        """Check if the index exists / is trained."""
        return self.is_trained

    def empty(self) -> None:
        self.is_trained = False

        for bucket in self.buckets.values():
            bucket.empty()

    def get_n_objects(self) -> int:
        return sum(map(Bucket.get_n_objects, self.buckets.values()))

    def get_total_capacity(self) -> int:
        return self.n_buckets * self.bucket_size

    def get_free_space(self) -> int:
        return self.get_total_capacity() - self.get_n_objects()

    def get_buckets(self) -> list[Bucket]:
        return list(self.buckets.values())
