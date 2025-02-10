from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dli.search_strategy import SearchStrategy


@dataclass
class SearchConfig:
    search_strategy: type[SearchStrategy]
    k: int
    nprobe: int
    python_max_workers: int | None
    faiss_max_threads: int | None
    verbose: bool

    def __post_init__(self) -> None:
        assert self.k > 0
        assert self.nprobe > 0

    def __str__(self) -> str:
        return f'SearchConfig(strategy={self.search_strategy.__name__}, k={self.k}, nprobe={self.nprobe})'
