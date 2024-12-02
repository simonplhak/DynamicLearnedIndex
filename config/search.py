from dataclasses import dataclass

from search_strategy import SearchStrategy


@dataclass
class SearchConfig:
    search_strategy: type[SearchStrategy]
    k: int
    nprobe: int

    def __post_init__(self) -> None:
        assert self.k > 0
        assert self.nprobe > 0
