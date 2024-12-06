from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, final, override

from faiss import METRIC_INNER_PRODUCT

from config import DatasetConfig, DistanceConfig, DLIConfig, ExperimentConfig, SearchConfig
from execution_environment.environment import Environment
from learned_index import LearnedMetricIndex
from search_strategy import KNNSearchStrategy, ModelDrivenSearchStrategy

if TYPE_CHECKING:
    from argparse import Namespace


@final
class Pro(Environment):
    @override
    def create_config(self, args: Namespace, commit_hash: str, dirty_state: bool) -> ExperimentConfig:
        return ExperimentConfig(
            DatasetConfig(
                dataset_size=10_000,
                X=Path('laion2B-en-clip768v2-n=300K.h5'),
                Q=Path('public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
                GT=Path('gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
            ),
            DLIConfig(
                LearnedMetricIndex,
                arity=3,
                bucket_shape=(200, 768),
                distance=DistanceConfig(METRIC_INNER_PRODUCT, keep_max=True),
                sample_threshold=100_000,
                compaction_strategy=args.compaction_strategy_class,
            ),
            [
                SearchConfig(
                    k=10,
                    search_strategy=search_strategy,
                    nprobe=nprobe,
                )
                for search_strategy in [KNNSearchStrategy, ModelDrivenSearchStrategy]
                for nprobe in [1, 5, 10]
            ],
            commit_hash,
            dirty_state,
        )
