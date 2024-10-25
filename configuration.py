from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from index import Index

type BucketShape = tuple[int, int]  # (number of vectors, dimensionality)


@dataclass
class DistanceConfig:
    metric: int
    """Which metric to use to compute the distance between objects."""
    keep_max: bool
    """Whether to keep the maximal or minimal values when computing the distance."""


@dataclass
class IndexConfig:
    n_buckets: int
    bucket_shape: BucketShape
    distance: DistanceConfig


@dataclass
class FrameworkConfig:
    index_class: type[Index]
    arity: int
    bucket_shape: BucketShape
    distance: DistanceConfig


@dataclass
class SearchConfig:
    k: int
    nprobe: int

    def __post_init__(self) -> None:
        assert self.k > 0
        assert self.nprobe > 0


@dataclass
class ExperimentConfig:
    framework_config: FrameworkConfig
    search_config: SearchConfig
