from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, stdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from config import SearchConfig
    from statistic import FrameworkCompactionStatistics, FrameworkSearchStatistics


@dataclass
class BuildResult:
    time: float
    stats: dict
    per_objects_insertion_statistics: list[FrameworkCompactionStatistics]

    def total_model_training_time(self) -> float:
        return sum(x.total_model_training_time for x in self.per_objects_insertion_statistics)

    def when_were_the_new_levels_allocated(self) -> list[int]:
        return [i for i, x in enumerate(self.per_objects_insertion_statistics) if x.allocated_new_level]

    def total_n_retrained_indexes(self) -> int:
        return sum(x.n_retrained_indexes for x in self.per_objects_insertion_statistics)


@dataclass
class ExperimentSearchResult:
    config: SearchConfig
    database_size: int
    n_queries: int
    sum_of_recalls: float
    n_candidates_per_query: Tensor
    total_search_time: float
    per_query_statistics: list[FrameworkSearchStatistics]

    def avg_recall(self) -> float:
        return self.sum_of_recalls / self.n_queries * 100

    # Number of candidates
    def min_n_candidates(self) -> int:
        return int(self.n_candidates_per_query.min())

    def mean_n_candidates(self) -> float:
        return float(self.n_candidates_per_query.mean())

    def median_n_candidates(self) -> float:
        return float(self.n_candidates_per_query.median())

    def max_n_candidates(self) -> int:
        return int(self.n_candidates_per_query.max())

    def std_n_candidates(self) -> float:
        return float(self.n_candidates_per_query.std())

    # Search time
    def min_search_time(self) -> float:
        return min(self.per_query_statistics, key=lambda x: x.total_search_time_in_ms).total_search_time_in_ms

    def mean_search_time(self) -> float:
        return mean(x.total_search_time_in_ms for x in self.per_query_statistics)

    def median_search_time(self) -> float:
        return median(x.total_search_time_in_ms for x in self.per_query_statistics)

    def max_search_time(self) -> float:
        return max(self.per_query_statistics, key=lambda x: x.total_search_time_in_ms).total_search_time_in_ms

    def std_search_time(self) -> float:
        return stdev(x.total_search_time_in_ms for x in self.per_query_statistics)

    # Other
    def candidates_percentage(self) -> float:
        return self.mean_n_candidates() / self.database_size * 100

    def avg_time_per_query_in_ms(self) -> float:
        return self.total_search_time / self.n_queries * 1_000

    def queries_per_second(self) -> float:
        return self.n_queries / self.total_search_time

    def get_stats(self) -> str:
        avg_recall = self.avg_recall()
        mean_n_candidates = self.mean_n_candidates()
        candidates_percentage = self.candidates_percentage()
        avg_time_per_query_in_ms = self.avg_time_per_query_in_ms()

        return (
            f'{avg_recall:.2f}%, '
            f'{mean_n_candidates:.2f} candidates ({candidates_percentage:.2f}%), '
            f'{avg_time_per_query_in_ms:.2f}ms per query'
        )
