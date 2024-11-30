import argparse
from argparse import Namespace

from compaction_strategy import compaction_strategies


def parse_arguments() -> Namespace:
    """Process command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--compaction-strategy', choices=compaction_strategies.keys(), required=True)
    args = parser.parse_args()

    compaction_strategy_class = compaction_strategies[args.compaction_strategy]

    return Namespace(compaction_strategy_class=compaction_strategy_class)
