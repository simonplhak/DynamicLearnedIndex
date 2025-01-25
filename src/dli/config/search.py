from dataclasses import dataclass

from dli.search_strategy import SearchStrategy


@dataclass
class SearchConfig:
    search_strategy: type[SearchStrategy]
    k: int
    nprobe: int

    def __post_init__(self) -> None:
        assert self.k > 0
        assert self.nprobe > 0

    def __str__(self) -> str:
        return f'SearchConfig(strategy={self.search_strategy.__name__}, k={self.k}, nprobe={self.nprobe})'
