from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework_compaction_statistics import FrameworkCompactionStatistics


@dataclass
class BuildResult:
    time: float
    stats: dict
    per_objects_insertion_statistics: list[FrameworkCompactionStatistics]

    def total_model_training_time(self) -> float:
        return sum(x.total_model_training_time for x in self.per_objects_insertion_statistics)
