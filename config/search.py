from dataclasses import dataclass


@dataclass
class SearchConfig:
    k: int
    nprobe: int

    def __post_init__(self) -> None:
        assert self.k > 0
        assert self.nprobe > 0
