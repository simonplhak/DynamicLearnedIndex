from enum import Enum

import faiss.contrib.torch_utils  # Wrapper allowing PyTorch tensors to be used with faiss
from faiss import Kmeans
from torch import Tensor, from_numpy

SEED = 42


class DistanceFunction(Enum):
    INNER_PRODUCT = faiss.METRIC_INNER_PRODUCT
    L2 = faiss.METRIC_L2


def knn(query: Tensor, data: Tensor, k: int, distance_function: DistanceFunction) -> tuple[Tensor, Tensor]:
    return faiss.knn(query, data, k, distance_function.value)  # type: ignore


def merge_knn_results(D_all: Tensor, I_all: Tensor, keep_max: bool) -> tuple[Tensor, Tensor]:  # noqa: FBT001
    D, I = faiss.merge_knn_results(D_all.numpy(), I_all.numpy(), keep_max)  # type: ignore
    return from_numpy(D), from_numpy(I)


def train_kmeans(data: Tensor, k: int) -> Kmeans:
    kmeans = faiss.Kmeans(
        d=data.shape[1],
        k=k,
        verbose=False,
        seed=SEED,
        spherical=True,
    )
    kmeans.train(data)
    return kmeans


def obtain_labels_via_kmeans(data: Tensor, k: int) -> Tensor:
    kmeans = train_kmeans(data, k)
    return kmeans.index.search(data, 1)[1].T[0]  # type: ignore
