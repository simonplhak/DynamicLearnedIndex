from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


def dfs_recursive(level: int, n_levels: int, arity: int, labels: list, counter: int) -> int:
    # TODO: refactor -- use stack
    if level >= n_levels:
        return counter

    for _ in range(arity):
        labels[level].append(counter)
        counter = dfs_recursive(level + 1, n_levels, arity, labels, counter + 1)

    return counter


def label_via_dfs(n_levels: int, arity: int) -> list[list[int]]:
    """Create labels based on a pre-order DFS traversal."""
    labels = [[] for _ in range(n_levels)]
    counter = 1

    labels[0].append(counter)
    dfs_recursive(1, n_levels, arity, labels, counter + 1)

    return labels


@dataclass
class SearchStrategy(ABC):
    arity: int
    n_levels: int

    @abstractmethod
    def determine_level_nprobe(self, level: int, overall_nprobe: int) -> int:
        raise NotImplementedError


@dataclass
class KNNSearchStrategy(SearchStrategy):
    def __post_init__(self):
        # Label nodes
        self.labels: list[list[int]] = label_via_dfs(self.n_levels, self.arity)

    def determine_level_nprobe(self, level: int, overall_nprobe: int) -> int:
        return len([x for x in self.labels[level] if x <= overall_nprobe])


@dataclass
class ModelDrivenSearchStrategy(SearchStrategy):
    def determine_level_nprobe(self, level: int, overall_nprobe: int) -> int:
        raise NotImplementedError  # ! Implemented in the search_model_driven method of the DynamicLearnedIndex class.


@dataclass
class LatestSearchStrategy(SearchStrategy):
    def determine_level_nprobe(self, level: int, overall_nprobe: int) -> int:
        raise NotImplementedError  # TODO: implement


@dataclass
class HistoricSearchStrategy(SearchStrategy):
    def determine_level_nprobe(self, level: int, overall_nprobe: int) -> int:
        raise NotImplementedError  # TODO: implement


if __name__ == '__main__':
    arity = 3
    n_levels = 3
    nprobe = 5

    print(label_via_dfs(n_levels, arity))

    print(f'{nprobe=}')
    for level in range(n_levels):
        print(f'{KNNSearchStrategy(arity, n_levels).determine_level_nprobe(level, nprobe)=}')
