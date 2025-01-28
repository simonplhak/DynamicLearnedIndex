from dataclasses import dataclass
from pathlib import Path

from datasets import DatasetConfig

from dli.config.dli import DLIConfig
from dli.config.search import SearchConfig


@dataclass
class ExperimentConfig:
    dataset_path_prefix: Path
    """Path prefix for the dataset on the given execution environment."""
    dataset_config: DatasetConfig
    """Configuration of the dataset."""
    dli_config: DLIConfig
    """Configuration of the DLI."""
    search_configs: list[SearchConfig]
    """Configurations of the search strategies."""

    # Reproducibility
    commit_hash: str
    """Hash of the current commit."""
    dirty_state: bool
    """Whether the repository was in a dirty state."""
