from __future__ import annotations

from math import ceil

import numpy as np
import torch
from torch import Tensor

from bucket import Bucket

SEED = 42
torch.manual_seed(SEED)


class DynamicBucket(Bucket):
    """A bucket that dynamically expands its capacity."""

    def __init__(self, bucket_shape: tuple[int, int], metric: int) -> None:
        super().__init__(bucket_shape, metric)

    def insert_single(self, X: Tensor, I: int) -> None:
        assert X.shape == (self.dimensionality,), 'X must be a 1D tensor'

        if self.n_objects + 1 > self.bucket_size:
            factor = self._calculate_resizing_factor(len(X))
            self._resize(factor)

        super().insert_single(X, I)

    def insert_bulk(self, X: Tensor, I: np.ndarray) -> None:
        if len(X) == 0:
            return

        assert len(X) == len(I), 'X and I must have the same length'

        if self.n_objects + len(X) > self.bucket_size:
            factor = self._calculate_resizing_factor(len(X))
            self._resize(factor)

        super().insert_bulk(X, I)

    def _resize(self, factor: int) -> None:
        """Resize the bucket to the given factor."""
        self.data = torch.cat((self.data, torch.zeros((self.bucket_size * (factor - 1), self.dimensionality))))
        self.ids = np.concatenate((self.ids, np.zeros(self.bucket_size * (factor - 1), dtype=np.int64)))
        self.bucket_size *= factor

    def _calculate_resizing_factor(self, new_n_objects: int) -> int:
        # We need to resize the bucket so that the existing objects (up to the self.bucket_size)
        # + the new objects (new_n_objects) will fit to the new bucket
        return ceil((new_n_objects + self.bucket_size) / self.bucket_size)
