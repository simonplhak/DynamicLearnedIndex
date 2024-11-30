from __future__ import annotations

from pathlib import Path
from typing import final, override

from faiss import METRIC_INNER_PRODUCT

from config.dataset import DatasetConfig
from config.distance import DistanceConfig
from config.dli import DLIConfig
from config.experiment import ExperimentConfig
from config.search import SearchConfig
from execution_environment.environment import Environment
from lmi import LMIIndex
from search_strategy import KNNSearchStrategy


@final
class Pro(Environment):
    @override
    def create_config(self, commit_hash: str, dirty_state: bool) -> ExperimentConfig:
        return ExperimentConfig(
            DatasetConfig(
                dataset_size=10_000,
                X=Path('laion2B-en-clip768v2-n=300K.h5'),
                Q=Path('public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
                GT=Path('gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
            ),
            DLIConfig(
                LMIIndex,
                arity=3,
                bucket_shape=(200, 768),
                distance=DistanceConfig(METRIC_INNER_PRODUCT, keep_max=True),
                sample_threshold=100_000,
                search_strategy=KNNSearchStrategy,
                # search_strategy=ModelDrivenSearchStrategy,
            ),
            [SearchConfig(k=10, nprobe=nprobe) for nprobe in [1, 2, 3, 4, 5, 10, 25, 50, 100]],
            commit_hash,
            dirty_state,
        )
