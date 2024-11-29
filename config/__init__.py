"""Environment configurations for the experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configuration import ExperimentConfig


def choose(hostname: str) -> ExperimentConfig:
    from config import david, metacentrum, pro

    match hostname:
        case 'Pro.local':
            return pro.experiment_config
        case name if name.startswith('david'):
            return david.experiment_config
        case _:
            return metacentrum.experiment_config
