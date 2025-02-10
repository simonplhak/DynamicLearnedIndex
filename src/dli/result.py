from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median, stdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from dli.config import SearchConfig
    from dli.statistic import FrameworkCompactionStatistics, FrameworkSearchStatistics


SEC_TO_MSEC = 1_000
TO_PERCENTAGE = 100


@dataclass
class BuildResult:
    time: float
    stats: dict
    per_objects_insertion_statistics: list[FrameworkCompactionStatistics] = field(repr=False)

    def total_model_training_time(self) -> float:
        return sum(x.total_model_training_time for x in self.per_objects_insertion_statistics)

    def when_were_the_new_levels_allocated(self) -> list[int]:
        return [i for i, x in enumerate(self.per_objects_insertion_statistics) if x.allocated_new_level]

    def total_n_retrained_indexes(self) -> int:
        return sum(x.n_retrained_indexes for x in self.per_objects_insertion_statistics)

    def total_deallocated_spaces(self) -> int:
        return sum(x.deallocated_spaces for x in self.per_objects_insertion_statistics)


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
        return self.sum_of_recalls / self.n_queries * TO_PERCENTAGE

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

    # Total search time
    def min_total_search_time(self) -> float:
        return min(self.per_query_statistics, key=lambda x: x.total_search_time_in_ms).total_search_time_in_ms

    def mean_total_search_time(self) -> float:
        return mean(x.total_search_time_in_ms for x in self.per_query_statistics)

    def median_total_search_time(self) -> float:
        return median(x.total_search_time_in_ms for x in self.per_query_statistics)

    def max_total_search_time(self) -> float:
        return max(self.per_query_statistics, key=lambda x: x.total_search_time_in_ms).total_search_time_in_ms

    def std_total_search_time(self) -> float:
        return stdev(x.total_search_time_in_ms for x in self.per_query_statistics)

    # Preparation time
    def min_preparation_time(self) -> float:
        return min(self.per_query_statistics, key=lambda x: x.preparation_time_in_ms).preparation_time_in_ms

    def mean_preparation_time(self) -> float:
        return mean(x.preparation_time_in_ms for x in self.per_query_statistics)

    def median_preparation_time(self) -> float:
        return median(x.preparation_time_in_ms for x in self.per_query_statistics)

    def max_preparation_time(self) -> float:
        return max(self.per_query_statistics, key=lambda x: x.preparation_time_in_ms).preparation_time_in_ms

    def std_preparation_time(self) -> float:
        return stdev(x.preparation_time_in_ms for x in self.per_query_statistics)

    # Search time
    def min_search_time(self) -> float:
        return min(self.per_query_statistics, key=lambda x: x.search_time_in_ms).search_time_in_ms

    def mean_search_time(self) -> float:
        return mean(x.search_time_in_ms for x in self.per_query_statistics)

    def median_search_time(self) -> float:
        return median(x.search_time_in_ms for x in self.per_query_statistics)

    def max_search_time(self) -> float:
        return max(self.per_query_statistics, key=lambda x: x.search_time_in_ms).search_time_in_ms

    def std_search_time(self) -> float:
        return stdev(x.search_time_in_ms for x in self.per_query_statistics)

    # Merge time
    def min_merge_time(self) -> float:
        return min(self.per_query_statistics, key=lambda x: x.merge_time_in_ms).merge_time_in_ms

    def mean_merge_time(self) -> float:
        return mean(x.merge_time_in_ms for x in self.per_query_statistics)

    def median_merge_time(self) -> float:
        return median(x.merge_time_in_ms for x in self.per_query_statistics)

    def max_merge_time(self) -> float:
        return max(self.per_query_statistics, key=lambda x: x.merge_time_in_ms).merge_time_in_ms

    def std_merge_time(self) -> float:
        return stdev(x.merge_time_in_ms for x in self.per_query_statistics)

    # Other
    def candidates_percentage(self) -> float:
        return self.mean_n_candidates() / self.database_size * TO_PERCENTAGE

    def avg_time_per_query_in_ms(self) -> float:
        return self.total_search_time / self.n_queries * SEC_TO_MSEC

    def queries_per_second(self) -> float:
        return self.n_queries / self.total_search_time

    def get_stats(self) -> str:
        def collect_statistics(name: str, method_prefix: str) -> str:
            min_time = getattr(self, f'min_{method_prefix}_time')()
            mean_time = getattr(self, f'mean_{method_prefix}_time')()
            max_time = getattr(self, f'max_{method_prefix}_time')()
            std_time = getattr(self, f'std_{method_prefix}_time')()

            return f'{name}: {min_time:>{25 - len(name)}.2f}ms, {mean_time:.2f}ms ±{std_time:.2f}ms, {max_time:.2f}ms'

        avg_recall = self.avg_recall()
        mean_n_candidates = self.mean_n_candidates()
        candidates_percentage = self.candidates_percentage()
        avg_time_per_query_in_ms = self.avg_time_per_query_in_ms()
        preparation_time_str = collect_statistics('Preparation time', 'preparation')
        search_time_str = collect_statistics('Search time', 'search')
        merge_time_str = collect_statistics('Merge time', 'merge')
        total_search_time_str = collect_statistics('Total search time', 'total_search')

        return (
            f'{avg_recall:.2f}%, '
            f'{mean_n_candidates:.2f} candidates ({candidates_percentage:.2f}%), '
            f'{avg_time_per_query_in_ms:.2f}ms per query\n'
            f'{preparation_time_str}\n'
            f'{search_time_str}\n'
            f'{merge_time_str}\n'
            f'-----------------\n'
            f'{total_search_time_str}\n'
        )
