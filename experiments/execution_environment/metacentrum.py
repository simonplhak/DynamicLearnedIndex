from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, final, override

from faiss import METRIC_INNER_PRODUCT

from dli.config import DatasetConfig, DistanceConfig, DLIConfig, ExperimentConfig, SearchConfig
from dli.learned_index import LearnedMetricIndex
from dli.search_strategy import KNNSearchStrategy, ModelDrivenSearchStrategy
from execution_environment.environment import Environment

if TYPE_CHECKING:
    from dli.cli import CLIArguments


PATH_PREFIX = Path('/storage/brno12-cerit/home/prochazka/fi-lmi-data/data/LAION2B')


@final
class Metacentrum(Environment):
    @override
    def create_config(self, args: CLIArguments, commit_hash: str, dirty_state: bool) -> ExperimentConfig:
        return ExperimentConfig(
            DatasetConfig(
                dataset_size=102_144_212,
                X=PATH_PREFIX / 'laion2B-en-clip768v2-n=100M.h5',
                Q=PATH_PREFIX / 'public-queries-2024-laion2B-en-clip768v2-n=10k.h5',
                GT=PATH_PREFIX / 'gold-standard-dbsize=100M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5',
            ),
            DLIConfig(
                LearnedMetricIndex,
                arity=3,
                bucket_shape=(5_000, 768),
                distance=DistanceConfig(METRIC_INNER_PRODUCT, keep_max=True),
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
                for nprobe in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            ],
            commit_hash,
            dirty_state,
        )
