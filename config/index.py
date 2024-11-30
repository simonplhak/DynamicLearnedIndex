from dataclasses import dataclass

from config.bucket_shape import BucketShape
from config.distance import DistanceConfig


@dataclass
class IndexConfig:
    n_buckets: int
    bucket_shape: BucketShape
    distance: DistanceConfig
    sample_threshold: int
    # Required to automatically determine the number of hidden neurons
    n_training_samples: int
