from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from pathlib import Path

    from configuration import ExperimentConfig
    from result import BuildResult, ExperimentSearchResult


def save_relevant_results_to_csv(
    config: ExperimentConfig,
    build_result: BuildResult,
    results: list[ExperimentSearchResult],
    experiment_dir: Path,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            'nprobe': [r.nprobe for r in config.search_configs],
            'avg_recall': [result.avg_recall() for result in results],
            'avg_time_per_query': [result.avg_time_per_query_in_ms() for result in results],
            'queries_per_second': [result.queries_per_second() for result in results],
            'build_time': build_result.time,
        },
    )
    df.to_csv(experiment_dir / 'df_plot.csv')
    return df


def plot_recall_vs_nprobe(df: pd.DataFrame, experiment_dir: Path) -> None:
    plt = sns.lineplot(x='nprobe', y='avg_recall', data=df)
    plt.set_title('Number of visited buckets vs. average recall')
    plt.set_xlabel('Number of visited buckets')
    plt.set_ylabel('Average recall')
    fig = plt.get_figure()
    assert fig
    fig.savefig(experiment_dir / 'recall_vs_nprobe.pdf')
    fig.clear()


def plot_recall_vs_avg_time_per_query(df: pd.DataFrame, experiment_dir: Path) -> None:
    plt = sns.lineplot(x='nprobe', y='avg_time_per_query', data=df)
    plt.set_title('Number of visited buckets vs. average time per query')
    plt.set_xlabel('Number of visited buckets')
    plt.set_ylabel('Average time per query')
    fig = plt.get_figure()
    assert fig
    fig.savefig(experiment_dir / 'queries_per_second_vs_recall.pdf')
    fig.clear()


def plot_queries_per_second_vs_recall(df: pd.DataFrame, experiment_dir: Path) -> None:
    plt = sns.lineplot(x='avg_recall', y='queries_per_second', data=df)
    plt.set_title('Average recall vs. queries per second')
    plt.set_xlabel('Average recall')
    plt.set_ylabel('Queries per second')
    fig = plt.get_figure()
    assert fig
    fig.savefig(experiment_dir / 'recall_vs_avg_time_per_query.pdf')
    fig.clear()


# TODO: histogram of time per query
# TODO: histogram of n_candidates
# TODO: histogram of recall per query
# TODO: histogram of bucket occupations?
