"""Compaction strategies."""

from dli.compaction_strategy.bentley_saxe import BentleySaxe
from dli.compaction_strategy.leveling import Leveling

compaction_strategies = {
    'bentley-saxe': BentleySaxe,
    'leveling': Leveling,
}
