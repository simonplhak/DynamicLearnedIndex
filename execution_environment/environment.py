from abc import ABC, abstractmethod

from config.experiment import ExperimentConfig


class Environment(ABC):
    @abstractmethod
    def create_config(self, commit_hash: str, dirty_state: bool) -> ExperimentConfig:  # noqa: FBT001
        raise NotImplementedError
