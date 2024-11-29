from __future__ import annotations

from pathlib import Path

from faiss import METRIC_INNER_PRODUCT

from configuration import DatasetConfig, DistanceConfig, ExperimentConfig, FrameworkConfig, SearchConfig
from lmi import LMIIndex
from main import commit_hash, dirty_state
from search_strategy import KNNSearchStrategy

experiment_config = ExperimentConfig(
    DatasetConfig(
        dataset_size=10_120_191,
        X=Path('laion2B-en-clip768v2-n=10M.h5'),
        Q=Path('public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
        GT=Path('gold-standard-dbsize=10M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
    ),
    FrameworkConfig(
        LMIIndex,
        arity=3,
        bucket_shape=(3_000, 768),
        distance=DistanceConfig(METRIC_INNER_PRODUCT, keep_max=True),
        sample_threshold=100_000,
        search_strategy=KNNSearchStrategy,
        # search_strategy=ModelDrivenSearchStrategy,
    ),
    [SearchConfig(k=30, nprobe=nprobe) for nprobe in [1, 2, 3, 4, 5, 10, 25, 50, 100]],
    commit_hash,
    dirty_state,
)
