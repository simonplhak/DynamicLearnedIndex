from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from index import Index

type BucketShape = tuple[int, int]  # (number of vectors, dimensionality)


@dataclass
class DatasetConfig:
    dataset_size: int
    """Number of objects in the dataset."""
    X: Path
    """Path to the dataset."""
    Q: Path
    """Path to the queries."""
    GT: Path
    """Path to the ground truth."""


@dataclass
class DistanceConfig:
    metric: int
    """Which metric to use to compute the distance between objects."""
    keep_max: bool
    """Whether to keep the maximal or minimal values when computing the distance."""


@dataclass
class SamplingConfig:
    percentage: float
    """Percentage of objects to sample. Must be between 0 and 1."""
    threshold: int
    """From which number of objects to start sampling. Must be at least 0."""

    def __post_init__(self) -> None:
        assert 0 <= self.percentage <= 1
        assert self.threshold >= 0


@dataclass
class IndexConfig:
    n_buckets: int
    bucket_shape: BucketShape
    distance: DistanceConfig
    sampling: SamplingConfig


@dataclass
class FrameworkConfig:
    index_class: type[Index]
    arity: int
    bucket_shape: BucketShape
    distance: DistanceConfig
    sampling: SamplingConfig


@dataclass
class SearchConfig:
    k: int
    nprobe: int

    def __post_init__(self) -> None:
        assert self.k > 0
        assert self.nprobe > 0


@dataclass
class ExperimentConfig:
    dataset_config: DatasetConfig
    framework_config: FrameworkConfig
    search_configs: list[SearchConfig]

    # Reproducibility
    commit_hash: str
    """Hash of the current commit."""
    dirty_state: bool
    """Whether the repository was in a dirty state."""
