from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameworkSearchStatistics:
    total_n_candidates: int
    n_candidates_per_level: list[int]
    search_time_per_level_in_ms: list[float]

    # Time measurements of all steps of the search method
    preparation_time_in_ms: float
    search_time_in_ms: float
    merge_time_in_ms: float

    total_search_time_in_ms: float

    # At what level is each result object located?
    result_object_level_location: list[tuple[int, int]]
