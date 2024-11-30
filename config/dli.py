from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compaction_strategy import BentleySaxe, Leveling
    from config.bucket_shape import BucketShape
    from config.distance import DistanceConfig
    from learned_index import LearnedIndex
    from search_strategy import SearchStrategy


@dataclass
class DLIConfig:
    # Construction
    index_class: type[LearnedIndex]
    arity: int
    bucket_shape: BucketShape
    distance: DistanceConfig
    sample_threshold: int
    compaction_strategy: type[BentleySaxe | Leveling]

    # Search
    search_strategy: type[SearchStrategy]
