from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from pathlib import Path

    from config import ExperimentConfig
    from result import BuildResult, ExperimentSearchResult


def save_relevant_results_to_csv(
    config: ExperimentConfig,
    build_result: BuildResult,
    results: list[ExperimentSearchResult],
    experiment_dir: Path,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            'search_strategy': [r.search_strategy.__name__ for r in config.search_configs],
            'nprobe': [r.nprobe for r in config.search_configs],
            'avg_recall': [result.avg_recall() for result in results],
            'avg_time_per_query': [result.avg_time_per_query_in_ms() for result in results],
            'queries_per_second': [result.queries_per_second() for result in results],
            'build_time': build_result.time,
        },
    )
    df.to_csv(experiment_dir / 'df_plot.csv')
    return df


def create_lineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Create and save a line plot using seaborn.

    Args:
        df: DataFrame containing the data
        x: Column name for x-axis
        y: Column name for y-axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_path: Path to save the plot

    """
    plt = sns.lineplot(x=x, y=y, hue='search_strategy', data=df)
    plt.set_title(title)
    plt.set_xlabel(xlabel)
    plt.set_ylabel(ylabel)
    fig = plt.get_figure()
    assert fig
    fig.savefig(output_path)
    fig.clear()


def plot_recall_vs_nprobe(df: pd.DataFrame, experiment_dir: Path) -> None:
    create_lineplot(
        df=df,
        x='nprobe',
        y='avg_recall',
        title='Number of visited buckets vs. average recall',
        xlabel='Number of visited buckets',
        ylabel='Average recall',
        output_path=experiment_dir / 'recall_vs_nprobe.pdf',
    )


def plot_recall_vs_avg_time_per_query(df: pd.DataFrame, experiment_dir: Path) -> None:
    create_lineplot(
        df=df,
        x='nprobe',
        y='avg_time_per_query',
        title='Number of visited buckets vs. average time per query',
        xlabel='Number of visited buckets',
        ylabel='Average time per query',
        output_path=experiment_dir / 'queries_per_second_vs_recall.pdf',
    )


def plot_queries_per_second_vs_recall(df: pd.DataFrame, experiment_dir: Path) -> None:
    create_lineplot(
        df=df,
        x='avg_recall',
        y='queries_per_second',
        title='Average recall vs. queries per second',
        xlabel='Average recall',
        ylabel='Queries per second',
        output_path=experiment_dir / 'recall_vs_avg_time_per_query.pdf',
    )


# TODO: histogram of time per query
# TODO: histogram of n_candidates
# TODO: histogram of recall per query
# TODO: histogram of bucket occupations?
