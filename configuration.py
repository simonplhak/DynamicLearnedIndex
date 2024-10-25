from __future__ import annotations

import subprocess
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
    sample_percentage: float


@dataclass
class FrameworkConfig:
    index_class: type[Index]
    arity: int
    bucket_shape: BucketShape
    distance: DistanceConfig
    sample_percentage: float


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
    search_configs: list[SearchConfig]

    commit_hash: str
    """Hash of the current commit."""
    dirty_state: bool
    """Whether the repository was in a dirty state."""

    def __init__(self, config: FrameworkConfig, search_configs: list[SearchConfig]) -> None:
        self.framework_config = config
        self.search_configs = search_configs

        self.commit_hash = subprocess.check_output(['git', 'describe', '--always']).strip().decode()  # noqa: S603, S607
        self.dirty_state = subprocess.call(['git', 'diff', '--quiet']) != 0  # noqa: S603, S607
