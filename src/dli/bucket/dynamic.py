from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, final, override

from torch import Tensor, cat, empty, float32, int64

from dli.bucket.bucket import Bucket

if TYPE_CHECKING:
    from dli.faiss_facade import DistanceFunction


@final
class DynamicBucket(Bucket):
    """A bucket that dynamically expands its capacity."""

    def __init__(
        self,
        bucket_shape: tuple[int, int],
        distance_function: DistanceFunction,
        shrink_buckets: bool,  # noqa: FBT001
    ) -> None:
        super().__init__(bucket_shape, distance_function, shrink_buckets)

    @override
    def insert_single(self, X: Tensor, I: int) -> None:
        assert X.shape == (self.dimensionality,), 'X must be a 1D tensor'

        # Solve the overflow
        if self.n_objects + 1 > self.bucket_size:
            factor = self._calculate_resizing_factor(len(X))
            self._resize(factor)

        self.data[self.n_objects] = X
        self.ids[self.n_objects] = I
        self.n_objects += 1

    @override
    def insert_bulk(self, X: Tensor, I: Tensor) -> None:
        if len(X) == 0:
            return

        assert len(X) == len(I), 'X and I must have the same length'

        # Solve the overflow
        if self.n_objects + len(X) > self.bucket_size:
            factor = self._calculate_resizing_factor(len(X))
            self._resize(factor)

        start, stop = self.n_objects, self.n_objects + len(X)

        self.data[start:stop] = X
        self.ids[start:stop] = I
        self.n_objects += len(X)

    def _resize(self, factor: int) -> None:
        """Resize the bucket to the given factor."""
        self.data = cat(
            (
                self.data,
                empty((self.bucket_size * (factor - 1), self.dimensionality), dtype=float32),
            ),
        )
        self.ids = cat(
            (
                self.ids,
                empty(self.bucket_size * (factor - 1), dtype=int64),
            ),
        )
        self.bucket_size *= factor

    def _calculate_resizing_factor(self, new_n_objects: int) -> int:
        # We need to resize the bucket so that the existing objects (up to the self.bucket_size)
        # + the new objects (new_n_objects) will fit to the new bucket
        return ceil((new_n_objects + self.bucket_size) / self.bucket_size)
