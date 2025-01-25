from abc import ABC, abstractmethod

from dli.cli import CLIArguments
from dli.config import ExperimentConfig


class Environment(ABC):
    @abstractmethod
    def create_config(self, args: CLIArguments, commit_hash: str, dirty_state: bool) -> ExperimentConfig:  # noqa: FBT001
        raise NotImplementedError
