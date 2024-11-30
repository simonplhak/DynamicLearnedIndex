from dataclasses import dataclass


@dataclass
class DistanceConfig:
    metric: int
    """Which metric to use to compute the distance between objects."""
    keep_max: bool
    """Whether to keep the maximal or minimal values when computing the distance."""
