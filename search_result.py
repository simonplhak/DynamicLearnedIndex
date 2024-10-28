from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from configuration import SearchConfig


@dataclass
class SearchResult:
    config: SearchConfig
    database_size: int
    n_queries: int
    recall_per_query: list[float]
    n_candidates_per_query: list[int]
    total_search_time: float

    def add(self, recall: float, n_candidates: int) -> None:
        self.recall_per_query.append(recall)
        self.n_candidates_per_query.append(n_candidates)

    def avg_recall(self) -> float:
        return sum(self.recall_per_query) / self.n_queries * 100

    def avg_n_candidates(self) -> float:
        return sum(self.n_candidates_per_query) / self.n_queries

    def candidates_percentage(self) -> float:
        return self.avg_n_candidates() / self.database_size * 100

    def avg_time_per_query_in_ms(self) -> float:
        return self.total_search_time / self.n_queries * 1_000

    def queries_per_second(self) -> float:
        return self.n_queries / self.total_search_time

    def log_stats(self) -> None:
        avg_recall = self.avg_recall()
        avg_n_candidates = self.avg_n_candidates()
        candidates_percentage = self.candidates_percentage()
        avg_time_per_query_in_ms = self.avg_time_per_query_in_ms()

        logger.info(
            f'{avg_recall:.2f}%, '
            f'{avg_n_candidates:.2f} candidates ({candidates_percentage:.2f}%), '
            f'{avg_time_per_query_in_ms:.2f}ms per query',
        )
