from __future__ import annotations

import numpy as np
import torch
from faiss import knn
from torch import Tensor

SEED = 42
torch.manual_seed(SEED)


class Bucket:
    def __init__(self, bucket_shape: tuple[int, int], metric: int) -> None:
        self.bucket_size, self.dimensionality = bucket_shape
        self.metric = metric

        self.data: Tensor = torch.zeros(bucket_shape)
        """Objects stored in the bucket."""
        self.ids: np.ndarray = np.zeros(self.bucket_size, dtype=np.int64)
        """Global index of each object in the bucket."""
        self.n_objects: int = 0
        """Current number of objects in the bucket."""

    def insert_single(self, X: Tensor, I: int) -> None:
        assert X.shape == (self.dimensionality,), 'X must be a 1D tensor'
        assert self.n_objects + 1 <= self.bucket_size, 'Bucket will overflow'

        self.data[self.n_objects] = X
        self.ids[self.n_objects] = I
        self.n_objects += 1

    def insert(self, X: Tensor, I: np.ndarray) -> None:
        if len(X) == 0:
            return

        # ! solve the issue of overflow in the caller... = keep the bucket abstraction simple
        assert self.n_objects + len(X) <= self.bucket_size, 'Bucket will overflow'
        assert len(X) == len(I), 'X and I must have the same length'

        start, stop = self.n_objects, self.n_objects + len(X)

        self.data[start:stop] = X
        self.ids[start:stop] = I
        self.n_objects += len(X)

    def search(self, query: Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert query.shape == (1, self.dimensionality)

        D, I = knn(query, self.data[: self.n_objects], k, metric=self.metric)

        return D[0], self.ids[I[0]]  # Convert local indexes back to the global IDs

    def empty(self) -> None:
        self.n_objects = 0

    def is_full(self) -> bool:
        return self.n_objects == self.bucket_size

    def is_empty(self) -> bool:
        return self.n_objects == 0

    def get_n_objects(self) -> int:
        return self.n_objects

    def get_data(self) -> Tensor:
        return self.data[: self.n_objects]

    def get_ids(self) -> np.ndarray:
        return self.ids[: self.n_objects]
