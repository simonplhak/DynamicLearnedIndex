from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, stdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configuration import SearchConfig
    from framework_search_statistics import FrameworkSearchStatistics


@dataclass
class ExperimentSearchResult:
    config: SearchConfig
    database_size: int
    n_queries: int
    recall_per_query: list[float]
    n_candidates_per_query: list[int]
    total_search_time: float
    per_query_statistics: list[FrameworkSearchStatistics]

    def add(self, recall: float, n_candidates: int) -> None:
        self.recall_per_query.append(recall)
        self.n_candidates_per_query.append(n_candidates)

    def avg_recall(self) -> float:
        return sum(self.recall_per_query) / self.n_queries * 100

    # Number of candidates
    def min_n_candidates(self) -> int:
        return min(self.n_candidates_per_query)

    def mean_n_candidates(self) -> float:
        return mean(self.n_candidates_per_query)

    def median_n_candidates(self) -> float:
        return median(self.n_candidates_per_query)

    def max_n_candidates(self) -> int:
        return max(self.n_candidates_per_query)

    def std_n_candidates(self) -> float:
        return stdev(self.n_candidates_per_query)

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
