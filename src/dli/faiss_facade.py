from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import faiss.contrib.torch_utils  # Wrapper allowing PyTorch tensors to be used with faiss
from faiss import Kmeans
from torch import Tensor, from_numpy

SEED = 42


@dataclass
class DistanceFunctionConfigMixin:
    distance_function: int
    """Which distance function to use to compute the distance between objects."""
    keep_max_values: bool
    """Whether to keep the maximal or minimal values when computing the distance."""
    normalize_centroids: bool
    """Whether to L2 normalize the centroids after each iteration during k-means training."""


class DistanceFunction(DistanceFunctionConfigMixin, Enum):
    INNER_PRODUCT = faiss.METRIC_INNER_PRODUCT, True, True
    L2 = faiss.METRIC_L2, False, False


def knn(query: Tensor, data: Tensor, k: int, distance_function: DistanceFunction) -> tuple[Tensor, Tensor]:
    return faiss.knn(query, data, k, distance_function.distance_function)  # type: ignore


def merge_knn_results(D_all: Tensor, I_all: Tensor, keep_max: bool) -> tuple[Tensor, Tensor]:  # noqa: FBT001
    D, I = faiss.merge_knn_results(D_all.numpy(), I_all.numpy(), keep_max)  # type: ignore
    return from_numpy(D), from_numpy(I)


def train_kmeans(data: Tensor, k: int, spherical: bool) -> Kmeans:  # noqa: FBT001
    kmeans = faiss.Kmeans(
        d=data.shape[1],
        k=k,
        verbose=False,
        seed=SEED,
        spherical=spherical,
    )
    kmeans.train(data)
    return kmeans


def obtain_labels_via_kmeans(data: Tensor, k: int, normalize_centroids: bool) -> Tensor:  # noqa: FBT001
    kmeans = train_kmeans(data, k, normalize_centroids)
    return kmeans.index.search(data, 1)[1].T[0]  # type: ignore
