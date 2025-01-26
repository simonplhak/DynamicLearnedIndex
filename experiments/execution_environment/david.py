from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, final, override

from dli.config import DatasetConfig, DLIConfig, ExperimentConfig, SearchConfig
from dli.faiss_facade import DistanceFunction
from dli.learned_index import LearnedMetricIndex
from dli.search_strategy import KNNSearchStrategy, ModelDrivenSearchStrategy
from execution_environment.environment import Environment

if TYPE_CHECKING:
    from dli.cli import CLIArguments


@final
class David(Environment):
    @override
    def create_config(self, args: CLIArguments, commit_hash: str, dirty_state: bool) -> ExperimentConfig:
        return ExperimentConfig(
            DatasetConfig(
                dataset_size=10_120_191,
                X=Path('laion2B-en-clip768v2-n=10M.h5'),
                Q=Path('public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
                GT=Path('gold-standard-dbsize=10M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5'),
            ),
            DLIConfig(
                LearnedMetricIndex,
                arity=3,
                bucket_shape=(3_000, 768),
                distance_function=DistanceFunction.INNER_PRODUCT,
                sample_threshold=100_000,
                compaction_strategy=args.compaction_strategy,
                shrink_buckets_during_compaction=args.shrink_buckets_during_compaction,
            ),
            [
                SearchConfig(
                    k=30,
                    search_strategy=search_strategy,
                    nprobe=nprobe,
                )
                for search_strategy in [KNNSearchStrategy, ModelDrivenSearchStrategy]
                for nprobe in [1, 2, 3, 4, 5, 10, 25, 50, 100]
            ],
            commit_hash,
            dirty_state,
        )
