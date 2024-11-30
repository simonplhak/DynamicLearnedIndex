from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compaction_strategy.bentley_saxe import BentleySaxe
    from compaction_strategy.leveling import Leveling
    from config import BucketShape
    from config.distance import DistanceConfig
    from internal_learned_index import InternalLearnedIndex
    from search_strategy import SearchStrategy


@dataclass
class DLIConfig:
    # Construction
    index_class: type[InternalLearnedIndex]
    arity: int
    bucket_shape: BucketShape
    distance: DistanceConfig
    sample_threshold: int
    compaction_strategy: type[BentleySaxe | Leveling]

    # Search
    search_strategy: type[SearchStrategy]
