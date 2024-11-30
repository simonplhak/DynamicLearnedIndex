from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    dataset_size: int
    """Number of objects in the dataset."""
    X: Path
    """Path to the dataset."""
    Q: Path
    """Path to the queries."""
    GT: Path
    """Path to the ground truth."""
