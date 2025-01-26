from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dli.compaction_strategy import BentleySaxe, Leveling
    from dli.config.bucket_shape import BucketShape
    from dli.faiss_facade import DistanceFunction
    from dli.learned_index import LearnedIndex


@dataclass
class DLIConfig:
    index_class: type[LearnedIndex]
    arity: int
    bucket_shape: BucketShape
    distance_function: DistanceFunction
    sample_threshold: int
    compaction_strategy: type[BentleySaxe | Leveling]
    shrink_buckets_during_compaction: bool
