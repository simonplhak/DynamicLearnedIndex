from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameworkCompactionStatistics:
    total_model_training_time: float
    total_compaction_time: float
