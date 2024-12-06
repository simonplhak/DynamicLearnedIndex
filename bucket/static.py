from __future__ import annotations

from typing import TYPE_CHECKING, final, override

from bucket.bucket import Bucket

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor


@final
class StaticBucket(Bucket):
    """A bucket with a fixed capacity."""

    def __init__(self, bucket_shape: tuple[int, int], metric: int, shrink_buckets: bool) -> None:  # noqa: FBT001
        super().__init__(bucket_shape, metric, shrink_buckets)

    @override
    def insert_single(self, X: Tensor, I: int) -> None:
        assert X.shape == (self.dimensionality,), 'X must be a 1D tensor'
        assert self.n_objects + 1 <= self.bucket_size, 'Bucket will overflow'

        self.data[self.n_objects] = X
        self.ids[self.n_objects] = I
        self.n_objects += 1

    @override
    def insert_bulk(self, X: Tensor, I: np.ndarray) -> None:
        if len(X) == 0:
            return

        # ! solve the issue of overflow in the caller... = keep the bucket abstraction simple
        assert self.n_objects + len(X) <= self.bucket_size, 'Bucket will overflow'
        assert len(X) == len(I), 'X and I must have the same length'

        start, stop = self.n_objects, self.n_objects + len(X)

        self.data[start:stop] = X
        self.ids[start:stop] = I
        self.n_objects += len(X)
