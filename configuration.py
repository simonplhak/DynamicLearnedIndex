from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from internal_learned_index import InternalLearnedIndex
    from search_strategy import SearchStrategy

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
class IndexConfig:
    n_buckets: int
    bucket_shape: BucketShape
    distance: DistanceConfig
    sample_threshold: int
    # Required to automatically determine the number of hidden neurons
    n_training_samples: int


@dataclass
class DLIConfig:
    index_class: type[InternalLearnedIndex]
    arity: int
    bucket_shape: BucketShape
    distance: DistanceConfig
    sample_threshold: int
    search_strategy: type[SearchStrategy]


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
    dli_config: DLIConfig
    search_configs: list[SearchConfig]

    # Reproducibility
    commit_hash: str
    """Hash of the current commit."""
    dirty_state: bool
    """Whether the repository was in a dirty state."""
