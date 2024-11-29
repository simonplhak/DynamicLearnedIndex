"""Environment configurations for the experiments."""

from config import david, metacentrum, pro
from configuration import ExperimentConfig


def choose_config(hostname: str, commit_hash: str, dirty_state: bool) -> ExperimentConfig:  # noqa: FBT001
    match hostname:
        case 'Pro.local':
            return pro.create_config(commit_hash, dirty_state)
        case name if name.startswith('david'):
            return david.create_config(commit_hash, dirty_state)
        case _:
            return metacentrum.create_config(commit_hash, dirty_state)
