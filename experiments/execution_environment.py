from __future__ import annotations

import socket
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, final, override

from cli import CLIArguments
from config import ExperimentConfig

from dli.config import DLIConfig, SearchConfig
from dli.faiss_facade import DistanceFunction
from dli.learned_index import LearnedMetricIndex
from dli.search_strategy import KNNSearchStrategy, ModelDrivenSearchStrategy

if TYPE_CHECKING:
    from experiments.cli import CLIArguments

METACENTRUM_DATASET_PATH_PREFIX = Path('/storage/brno12-cerit/home/prochazka/fi-lmi-data/data/LAION2B')
PRO_DATASET_PATH_PREFIX = Path('./data')


def detect_environment() -> Environment:
    match socket.gethostname():
        case 'Pro.local':
            return Pro()
        case _:
            return Metacentrum()


class Environment(ABC):
    @abstractmethod
    def create_config(self, args: CLIArguments, commit_hash: str, dirty_state: bool) -> ExperimentConfig:  # noqa: FBT001
        raise NotImplementedError


@final
class Metacentrum(Environment):
    @override
    def create_config(self, args: CLIArguments, commit_hash: str, dirty_state: bool) -> ExperimentConfig:
        return ExperimentConfig(
            METACENTRUM_DATASET_PATH_PREFIX,
            args.dataset_config,
            DLIConfig(
                LearnedMetricIndex,
                arity=3,
                bucket_shape=(5_000, 768),
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
                    faiss_max_threads=2,
                    python_max_workers=4,
                    verbose=False,
                )
                for search_strategy in [KNNSearchStrategy, ModelDrivenSearchStrategy]
                for nprobe in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            ],
            commit_hash,
            dirty_state,
        )


@final
class Pro(Environment):
    @override
    def create_config(self, args: CLIArguments, commit_hash: str, dirty_state: bool) -> ExperimentConfig:
        return ExperimentConfig(
            PRO_DATASET_PATH_PREFIX,
            args.dataset_config,
            DLIConfig(
                LearnedMetricIndex,
                arity=3,
                bucket_shape=(200, 768),
                distance_function=DistanceFunction.INNER_PRODUCT,
                sample_threshold=5_000,
                compaction_strategy=args.compaction_strategy,
                shrink_buckets_during_compaction=args.shrink_buckets_during_compaction,
            ),
            [
                SearchConfig(
                    k=10,
                    search_strategy=search_strategy,
                    nprobe=nprobe,
                    faiss_max_threads=None,
                    python_max_workers=None,
                    verbose=True,
                )
                for search_strategy in [KNNSearchStrategy, ModelDrivenSearchStrategy]
                for nprobe in [1, 5, 10]
            ],
            commit_hash,
            dirty_state,
        )
