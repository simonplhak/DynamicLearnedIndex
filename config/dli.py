from dataclasses import dataclass

from config import BucketShape
from config.distance import DistanceConfig
from internal_learned_index import InternalLearnedIndex
from search_strategy import SearchStrategy


@dataclass
class DLIConfig:
    index_class: type[InternalLearnedIndex]
    arity: int
    bucket_shape: BucketShape
    distance: DistanceConfig
    sample_threshold: int
    search_strategy: type[SearchStrategy]
