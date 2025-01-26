from dataclasses import dataclass

from dli.faiss_facade import DistanceFunction


@dataclass
class DistanceConfig:
    distance_function: DistanceFunction
    """Which metric to use to compute the distance between objects."""
    keep_max: bool
    """Whether to keep the maximal or minimal values when computing the distance."""
