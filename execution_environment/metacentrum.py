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

PATH_PREFIX = Path('/storage/brno12-cerit/home/prochazka/fi-lmi-data/data/LAION2B')


@final
class Metacentrum(Environment):
    @override
    def create_config(self, commit_hash: str, dirty_state: bool) -> ExperimentConfig:
        return ExperimentConfig(
            DatasetConfig(
                dataset_size=102_144_212,
                X=PATH_PREFIX / 'laion2B-en-clip768v2-n=100M.h5',
                Q=PATH_PREFIX / 'public-queries-2024-laion2B-en-clip768v2-n=10k.h5',
                GT=PATH_PREFIX / 'gold-standard-dbsize=100M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5',
            ),
            DLIConfig(
                LMIIndex,
                arity=3,
                bucket_shape=(5_000, 768),
                distance=DistanceConfig(METRIC_INNER_PRODUCT, keep_max=True),
                sample_threshold=100_000,
                search_strategy=KNNSearchStrategy,
                # search_strategy=ModelDrivenSearchStrategy,
            ),
            [SearchConfig(k=30, nprobe=nprobe) for nprobe in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]],
            commit_hash,
            dirty_state,
        )
