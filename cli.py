from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass

from compaction_strategy import BentleySaxe, Leveling, compaction_strategies


@dataclass
class CLIArguments:
    compaction_strategy: type[BentleySaxe | Leveling]
    shrink_buckets_during_compaction: bool

    def __init__(self, args: Namespace) -> None:
        self.compaction_strategy = compaction_strategies[args.compaction_strategy]
        self.shrink_buckets_during_compaction = args.shrink_buckets_during_compaction


def parse_arguments() -> CLIArguments:
    """Process command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('--compaction-strategy', choices=compaction_strategies.keys(), required=True)
    parser.add_argument('--shrink-buckets-during-compaction', type=bool, required=True)
    args = parser.parse_args()

    return CLIArguments(args)
