from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from faiss import knn
from torch import Tensor


class Bucket(ABC):
    """An abstract class representing a bucket."""

    def __init__(self, bucket_shape: tuple[int, int], metric: int, shrink_buckets: bool) -> None:  # noqa: FBT001
        self.default_bucket_size = bucket_shape[0]
        """The default size of the bucket before resizing."""
        self.bucket_size, self.dimensionality = bucket_shape
        self.metric = metric
        self.shrink_buckets: bool = shrink_buckets
        """Whether to shrink the buckets when calling the empty method."""

        self.data: Tensor = torch.empty(bucket_shape, dtype=torch.float32)
        """Objects stored in the bucket."""
        self.ids: np.ndarray = np.empty(self.bucket_size, dtype=np.int64)
        """Global index of each object in the bucket."""
        self.n_objects: int = 0
        """Current number of objects in the bucket."""

    @abstractmethod
    def insert_single(self, X: Tensor, I: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert_bulk(self, X: Tensor, I: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, query: Tensor, k: int, nprobe: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Search for the k nearest neighbors of the given query in the bucket.

        Parameters
        ----------
        query : Tensor
            Single query vector of shape (1, dimensionality).
        k : int
            Number of nearest neighbors to search for.
        nprobe : int
            Number of buckets to probe at each level.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, int]
            A tuple containing the neighbor distances, neighbor indices, and the size of the candidate set.

        """
        _ = nprobe  # We do not use nprobe for a single bucket

        assert query.shape == (1, self.dimensionality)

        D, I = knn(query, self.data[: self.n_objects], k, metric=self.metric)
        n_candidates = self.n_objects

        return D[0], self.ids[I[0]], n_candidates  # Convert local indexes back to the global IDs

    def empty(self) -> int:
        """Empty the bucket by setting the number of objects to 0.

        When shrink_buckets is True, the data and ID tensors are resized to the original shape.

        Returns
        -------
        int
            The difference between the current bucket size and the original bucket size.

        """
        self.n_objects = 0

        if self.shrink_buckets and self.bucket_size != self.default_bucket_size:  # The bucket has been resized
            del self.data, self.ids

            assert self.bucket_size > self.default_bucket_size

            deallocated_spaces = self.bucket_size - self.default_bucket_size

            self.bucket_size = self.default_bucket_size
            self.data = torch.empty((self.bucket_size, self.dimensionality), dtype=torch.float32)
            self.ids = np.empty(self.bucket_size, dtype=np.int64)

            return deallocated_spaces

        return 0

    def is_full(self) -> bool:
        return self.n_objects == self.bucket_size

    def is_empty(self) -> bool:
        return self.n_objects == 0

    def get_n_objects(self) -> int:
        return self.n_objects

    def get_capacity(self) -> int:
        return self.bucket_size

    def get_free_space(self) -> int:
        return self.bucket_size - self.n_objects

    def get_data(self) -> Tensor:
        return self.data[: self.n_objects]

    def get_ids(self) -> np.ndarray:
        return self.ids[: self.n_objects]

    def get_allocated_memory(self) -> int:
        data_memory = self.data.element_size() * self.data.nelement()
        ids_memory = self.ids.nbytes

        return data_memory + ids_memory
