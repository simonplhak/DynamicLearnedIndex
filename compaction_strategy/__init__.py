"""Compaction strategies."""

from compaction_strategy.bentley_saxe import BentleySaxe
from compaction_strategy.leveling import Leveling

compaction_strategies = {
    'bentley-saxe': BentleySaxe,
    'leveling': Leveling,
}
