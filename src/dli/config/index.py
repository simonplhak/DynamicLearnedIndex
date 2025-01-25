from dataclasses import dataclass

from dli.config.bucket_shape import BucketShape
from dli.config.distance import DistanceConfig


@dataclass
class IndexConfig:
    n_buckets: int
    bucket_shape: BucketShape
    distance: DistanceConfig
    sample_threshold: int
    shrink_buckets_during_compaction: bool
    # Required to automatically determine the number of hidden neurons
    n_training_samples: int
