from dataclasses import dataclass

from dli.config.dataset import DatasetConfig
from dli.config.dli import DLIConfig
from dli.config.search import SearchConfig


@dataclass
class ExperimentConfig:
    dataset_config: DatasetConfig
    dli_config: DLIConfig
    search_configs: list[SearchConfig]

    # Reproducibility
    commit_hash: str
    """Hash of the current commit."""
    dirty_state: bool
    """Whether the repository was in a dirty state."""
